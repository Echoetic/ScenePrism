import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
    'batch_size': 96,           # 优化的batch size
    'lr': 0.001,
    'epochs': 40,               # 增加训练轮数
    'image_size': 224,
    'seed': 42,
    'num_classes': 6,
    'num_workers': 16,
    'class_map': {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5},
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,
    'use_ensemble': True,       # 使用模型集成
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

print(f"=" * 70)
print(f"硬件配置信息")
print(f"=" * 70)
print(f"设备: {CONFIG['device']}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA版本: {torch.version.cuda}")
print(f"PyTorch版本: {torch.__version__}")
print(f"=" * 70)

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
            print(f"图片读取失败: {path}")
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
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

# ================= 多模型架构 =================

class ResNet50Classifier(nn.Module):
    """ResNet50分类器"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetClassifier(nn.Module):
    """EfficientNet分类器"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.efficientnet_b3(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class DenseNetClassifier(nn.Module):
    """DenseNet分类器"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.densenet121(pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_model(model_name='resnet50'):
    """获取指定模型"""
    if model_name == 'resnet50':
        return ResNet50Classifier(CONFIG['num_classes'])
    elif model_name == 'efficientnet':
        return EfficientNetClassifier(CONFIG['num_classes'])
    elif model_name == 'densenet':
        return DenseNetClassifier(CONFIG['num_classes'])
    else:
        raise ValueError(f"未知模型: {model_name}")

# ================= 训练函数 =================

def train_single_model(model_name, train_loader, val_loader):
    """训练单个模型"""
    print(f"\n{'='*70}")
    print(f"训练模型: {model_name}")
    print(f"{'='*70}")
    
    model = get_model(model_name).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] else None
    
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{CONFIG["epochs"]} [训练]', leave=False)
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
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # 验证
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
        scheduler.step()
        
        print(f"Epoch {epoch+1:3d} | 训练 Acc: {train_acc:5.2f}% | 验证 Acc: {val_acc:5.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }, f"{CONFIG['model_save_dir']}/best_{model_name}.pth")
            print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停: 验证准确率已连续 {patience} 轮未提升")
                break
    
    return best_val_acc

def train():
    """主训练流程"""
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
    
    # 训练多个模型
    if CONFIG['use_ensemble']:
        model_names = ['resnet50', 'efficientnet', 'densenet']
    else:
        model_names = ['resnet50']
    
    results = {}
    for model_name in model_names:
        best_acc = train_single_model(model_name, train_loader, val_loader)
        results[model_name] = best_acc
    
    # 测试最佳模型
    print(f"\n{'='*70}")
    print("测试集评估")
    print(f"{'='*70}")
    
    for model_name in model_names:
        model = get_model(model_name).to(CONFIG['device'])
        checkpoint = torch.load(f"{CONFIG['model_save_dir']}/best_{model_name}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f"{model_name:15s} | 验证: {results[model_name]:5.2f}% | 测试: {test_acc:5.2f}%")
    
    print(f"{'='*70}")

# ================= 预测函数 =================

def predict():
    """集成预测"""
    print(f"\n{'='*70}")
    print("模型预测")
    print(f"{'='*70}")
    
    # 获取可用模型
    if CONFIG['use_ensemble']:
        model_names = ['resnet50', 'efficientnet', 'densenet']
        available_models = [m for m in model_names if os.path.exists(f"{CONFIG['model_save_dir']}/best_{m}.pth")]
    else:
        available_models = ['resnet50'] if os.path.exists(f"{CONFIG['model_save_dir']}/best_resnet50.pth") else []
    
    if not available_models:
        print("错误: 未找到训练好的模型")
        return
    
    print(f"使用模型: {', '.join(available_models)}")
    
    # 加载模型
    models_list = []
    for model_name in available_models:
        model = get_model(model_name).to(CONFIG['device'])
        checkpoint = torch.load(f"{CONFIG['model_save_dir']}/best_{model_name}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models_list.append(model)
        print(f"  ✓ {model_name} (验证准确率: {checkpoint['val_acc']:.2f}%)")
    
    # 获取预测图片
    pred_imgs = [os.path.join(CONFIG['pred_root'], i) 
                 for i in os.listdir(CONFIG['pred_root']) 
                 if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"\n待预测图片: {len(pred_imgs)} 张")
    
    dataset = SceneDataset(pred_imgs, mode='pred', transform=get_transforms('val'))
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # 集成预测
    results = []
    with torch.no_grad():
        for images, filenames in tqdm(loader, desc='预测进度'):
            images = images.to(CONFIG['device'])
            
            # 多模型投票
            all_preds = []
            for model in models_list:
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.append(predicted.cpu().numpy())
            
            # 投票或平均
            if len(all_preds) > 1:
                all_preds = np.array(all_preds)
                final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, all_preds)
            else:
                final_preds = all_preds[0]
            
            for fname, pred in zip(filenames, final_preds):
                try:
                    pic_num = int(os.path.splitext(fname)[0])
                except ValueError:
                    pic_num = fname
                
                results.append({'pic_num': pic_num, 'predict_label': int(pred)})
    
    # 保存结果
    df = pd.DataFrame(results)
    df = df.sort_values(by='pic_num').reset_index(drop=True)
    df.to_excel(CONFIG['save_path'], index=False)
    
    print(f"\n✓ 预测完成! 结果已保存至: {CONFIG['save_path']}")
    print(f"{'='*70}")

def predict_new_images(test_image_dir, output_path='./test_result.xlsx'):
    """验收测试接口"""
    CONFIG['pred_root'] = test_image_dir
    CONFIG['save_path'] = output_path
    predict()

if __name__ == "__main__":
    train()
    predict()