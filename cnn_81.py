import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ================= 配置参数 =================
CONFIG = {
    'data_root': './data/basic_data',
    'pred_root': './data/pred_data',
    'save_path': './pred_result.xlsx',
    'model_save_dir': './models',
    'batch_size': 128,          # CNN相对轻量,可以用更大batch
    'lr': 0.001,
    'epochs': 40,
    'image_size': 224,
    'num_classes': 6,
    'num_workers': 16,
    'seed': 42,
    'class_map': {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5},
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(CONFIG['seed'])
os.makedirs(CONFIG['model_save_dir'], exist_ok=True)

print(f"{'='*70}")
print(f"自定义CNN场景分类模型")
print(f"{'='*70}")
print(f"设备: {CONFIG['device']}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"{'='*70}")

# ================= 数据集 =================

class SceneDataset(Dataset):
    def __init__(self, file_paths, labels=None, mode='train', transform=None):
        super().__init__()
        self.file_paths = file_paths
        self.labels = labels
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        path = self.file_paths[index]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (150, 150), color='white')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'pred':
            return image, os.path.basename(path)
        else:
            return image, self.labels[index]

    def __len__(self):
        return len(self.file_paths)

def get_transforms(mode='train'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.3)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

# ================= CNN模型架构 =================

class ConvBlock(nn.Module):
    """卷积块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ImprovedCNN(nn.Module):
    """
    改进的CNN模型,结合了多种现代技术:
    - 残差连接 (ResNet)
    - Squeeze-and-Excitation注意力机制
    - 全局平均池化
    - Dropout正则化
    """
    def __init__(self, num_classes=6):
        super().__init__()
        
        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 残差块组
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.se1 = SEBlock(128)
        
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.se2 = SEBlock(256)
        
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.se3 = SEBlock(512)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始特征提取
        x = self.init_conv(x)
        
        # 残差块 + SE注意力
        x = self.layer1(x)
        x = self.se1(x)
        
        x = self.layer2(x)
        x = self.se2(x)
        
        x = self.layer3(x)
        x = self.se3(x)
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class SimpleCNN(nn.Module):
    """
    简化版CNN模型,适合快速训练和测试
    """
    def __init__(self, num_classes=6):
        super().__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            ConvBlock(3, 64, pool=True),
            
            # Block 2: 64 -> 128
            ConvBlock(64, 128, pool=True),
            
            # Block 3: 128 -> 256
            ConvBlock(128, 256, pool=True),
            
            # Block 4: 256 -> 512
            ConvBlock(256, 512, pool=True),
            
            # Block 5: 512 -> 512
            ConvBlock(512, 512, pool=True),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(model_type='improved'):
    """
    创建模型
    model_type: 'improved' 或 'simple'
    """
    if model_type == 'improved':
        model = ImprovedCNN(num_classes=CONFIG['num_classes'])
    else:
        model = SimpleCNN(num_classes=CONFIG['num_classes'])
    return model

# ================= 训练函数 =================

def train(model_type='improved'):
    print(f"\n{'='*70}")
    print("数据加载")
    print(f"{'='*70}")
    
    # 加载数据
    all_paths = []
    all_labels = []
    for cls_name, idx in CONFIG['class_map'].items():
        cls_folder = os.path.join(CONFIG['data_root'], cls_name)
        if os.path.exists(cls_folder):
            imgs = [os.path.join(cls_folder, i) for i in os.listdir(cls_folder) 
                   if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
            all_paths.extend(imgs)
            all_labels.extend([idx] * len(imgs))
            print(f"  {cls_name:12s} ({idx}): {len(imgs):5d} 张")
    
    print(f"  {'总计':12s}     : {len(all_paths):5d} 张")
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=CONFIG['seed']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=CONFIG['seed']
    )
    
    print(f"\n数据集划分: 训练 {len(X_train)} | 验证 {len(X_val)} | 测试 {len(X_test)}")
    
    # 数据加载器
    train_dataset = SceneDataset(X_train, y_train, 'train', get_transforms('train'))
    val_dataset = SceneDataset(X_val, y_val, 'val', get_transforms('val'))
    test_dataset = SceneDataset(X_test, y_test, 'test', get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                             num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                           num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    
    print(f"\n{'='*70}")
    print(f"构建CNN模型 (类型: {model_type})")
    print(f"{'='*70}")
    
    model = get_model(model_type).to(CONFIG['device'])
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器和学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG['lr'], 
        steps_per_epoch=len(train_loader), 
        epochs=CONFIG['epochs']
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] else None
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print("开始训练")
    print(f"{'='*70}")
    
    for epoch in range(CONFIG['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["epochs"]} [训练]', leave=False)
        for images, labels in train_bar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            
            if CONFIG['mixed_precision']:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step()  # OneCycleLR需要每个batch更新
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1:3d} | 训练 Acc: {train_acc:5.2f}% | 验证 Acc: {val_acc:5.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }, f"{CONFIG['model_save_dir']}/best_cnn.pth")
            print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停: 验证准确率已连续 {patience} 轮未提升")
                break
    
    # 测试阶段
    print(f"\n{'='*70}")
    print("测试集评估")
    print(f"{'='*70}")
    
    checkpoint = torch.load(f"{CONFIG['model_save_dir']}/best_cnn.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='测试进度'):
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"\n最终结果:")
    print(f"  验证集准确率: {best_val_acc:.2f}%")
    print(f"  测试集准确率: {test_acc:.2f}%")
    print(f"{'='*70}")

# ================= 预测函数 =================

def predict(model_type='improved'):
    print(f"\n{'='*70}")
    print("模型预测")
    print(f"{'='*70}")
    
    model_path = f"{CONFIG['model_save_dir']}/best_cnn.pth"
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return
    
    model = get_model(model_type).to(CONFIG['device'])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"已加载模型 (验证准确率: {checkpoint['val_acc']:.2f}%)")
    
    pred_imgs = [os.path.join(CONFIG['pred_root'], i) 
                 for i in os.listdir(CONFIG['pred_root']) 
                 if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"找到 {len(pred_imgs)} 张待预测图片")
    
    dataset = SceneDataset(pred_imgs, mode='pred', transform=get_transforms('val'))
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    results = []
    with torch.no_grad():
        for images, filenames in tqdm(loader, desc='预测进度'):
            images = images.to(CONFIG['device'])
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for fname, pred in zip(filenames, predicted):
                try:
                    pic_num = int(os.path.splitext(fname)[0])
                except ValueError:
                    pic_num = fname
                
                results.append({'pic_num': pic_num, 'predict_label': pred.item()})
    
    df = pd.DataFrame(results)
    df = df.sort_values(by='pic_num').reset_index(drop=True)
    df.to_excel(CONFIG['save_path'], index=False)
    
    print(f"\n✓ 预测完成! 结果已保存至: {CONFIG['save_path']}")
    print(f"{'='*70}")

def predict_new_images(test_image_dir, output_path='./test_result.xlsx', model_type='improved'):
    """验收测试接口"""
    CONFIG['pred_root'] = test_image_dir
    CONFIG['save_path'] = output_path
    predict(model_type)

if __name__ == "__main__":
    # 选择模型类型: 'improved' (推荐) 或 'simple' (快速)
    MODEL_TYPE = 'improved'
    
    train(MODEL_TYPE)
    predict(MODEL_TYPE)