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

# ================= 1. 配置参数 =================
CONFIG = {
    'data_root': './data/basic_data',
    'pred_root': './data/pred_data',
    'save_path': './pred_result.xlsx',
    'model_save_dir': './models',
    'batch_size': 128,          # 利用32GB显存,大幅提升batch size
    'lr': 0.001,
    'epochs': 30,               # 增加训练轮数
    'image_size': 224,
    'seed': 42,
    'num_classes': 6,
    'num_workers': 16,          # 利用22核CPU加速数据加载
    'class_map': {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5},
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,    # 启用混合精度训练
    'scheduler_step': 10,       # 学习率调整步长
    'scheduler_gamma': 0.1,     # 学习率衰减因子
}

def set_seed(seed):
    """设置随机种子确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 启用cudnn加速

set_seed(CONFIG['seed'])
os.makedirs(CONFIG['model_save_dir'], exist_ok=True)

print(f"使用设备: {CONFIG['device']}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ================= 2. 数据集类 =================

class SceneDataset(Dataset):
    """场景图像数据集"""
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
            print(f"读取图片失败: {path}, 错误: {e}")
            # 返回一个空白图像作为fallback
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
    """数据增强和预处理"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

# ================= 3. 模型构建 =================

class SceneClassifier(nn.Module):
    """基于ResNet50的场景分类器"""
    def __init__(self, num_classes=6, pretrained=True):
        super(SceneClassifier, self).__init__()
        # 使用预训练的ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 替换最后的全连接层
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

def get_model():
    """创建模型"""
    model = SceneClassifier(num_classes=CONFIG['num_classes'], pretrained=True)
    return model

# ================= 4. 训练逻辑 =================

def train():
    """训练模型"""
    print("=" * 60)
    print("开始加载数据...")
    print("=" * 60)
    
    # 加载所有图片路径和标签
    all_paths = []
    all_labels = []
    for cls_name, idx in CONFIG['class_map'].items():
        cls_folder = os.path.join(CONFIG['data_root'], cls_name)
        if os.path.exists(cls_folder):
            imgs = [os.path.join(cls_folder, i) for i in os.listdir(cls_folder) 
                   if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
            all_paths.extend(imgs)
            all_labels.extend([idx] * len(imgs))
            print(f"类别 {cls_name} ({idx}): {len(imgs)} 张图片")
    
    print(f"\n总计: {len(all_paths)} 张图片")
    
    # 划分数据集: 80% 训练, 10% 验证, 10% 测试
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=CONFIG['seed']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=CONFIG['seed']
    )
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(X_train)} 张")
    print(f"验证集: {len(X_val)} 张")
    print(f"测试集: {len(X_test)} 张")
    
    # 创建数据加载器
    train_dataset = SceneDataset(X_train, y_train, 'train', get_transforms('train'))
    val_dataset = SceneDataset(X_val, y_val, 'val', get_transforms('val'))
    test_dataset = SceneDataset(X_test, y_test, 'test', get_transforms('val'))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    print("\n" + "=" * 60)
    print("构建模型...")
    print("=" * 60)
    
    # 创建模型
    model = get_model().to(CONFIG['device'])
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=CONFIG['scheduler_step'], 
        gamma=CONFIG['scheduler_gamma']
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] else None
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    for epoch in range(CONFIG['epochs']):
        # ============ 训练阶段 ============
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["epochs"]} [训练]')
        for images, labels in train_bar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # 混合精度训练
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
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # ============ 验证阶段 ============
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{CONFIG["epochs"]} [验证]')
            for images, labels in val_bar:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        
        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印epoch总结
        print(f"\nEpoch {epoch+1} 总结:")
        print(f"  训练集 - Loss: {train_loss/train_total:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证集 - Loss: {val_loss/val_total:.4f}, Acc: {val_acc:.2f}%")
        print(f"  学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f"{CONFIG['model_save_dir']}/best_model.pth")
            print(f"  >>> 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        print("-" * 60)
    
    # ============ 测试阶段 ============
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型...")
    print("=" * 60)
    
    # 加载最佳模型
    checkpoint = torch.load(f"{CONFIG['model_save_dir']}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='测试集评估')
        for images, labels in test_bar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            test_bar.set_postfix({'acc': f'{100.*test_correct/test_total:.2f}%'})
    
    test_acc = 100. * test_correct / test_total
    
    print(f"\n最终结果:")
    print(f"  最佳模型来自 Epoch {best_epoch}")
    print(f"  验证集准确率: {best_val_acc:.2f}%")
    print(f"  测试集准确率: {test_acc:.2f}%")
    print("=" * 60)

# ================= 5. 预测导出 =================

def predict():
    """预测并导出结果到Excel"""
    print("\n" + "=" * 60)
    print("开始预测...")
    print("=" * 60)
    
    # 检查模型文件
    model_path = f"{CONFIG['model_save_dir']}/best_model.pth"
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        print("请先运行 train() 训练模型")
        return
    
    # 加载模型
    model = get_model().to(CONFIG['device'])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"已加载模型 (验证准确率: {checkpoint['val_acc']:.2f}%)")
    
    # 获取预测图片
    pred_imgs = [os.path.join(CONFIG['pred_root'], i) 
                 for i in os.listdir(CONFIG['pred_root']) 
                 if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(pred_imgs) == 0:
        print(f"错误: 在 {CONFIG['pred_root']} 中未找到图片")
        return
    
    print(f"找到 {len(pred_imgs)} 张待预测图片")
    
    # 创建数据集和加载器
    dataset = SceneDataset(pred_imgs, mode='pred', transform=get_transforms('val'))
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # 预测
    results = []
    with torch.no_grad():
        pred_bar = tqdm(loader, desc='预测进度')
        for images, filenames in pred_bar:
            images = images.to(CONFIG['device'])
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for fname, pred in zip(filenames, predicted):
                # 解析文件名
                try:
                    pic_num = int(os.path.splitext(fname)[0])
                except ValueError:
                    pic_num = fname
                
                results.append({
                    'pic_num': pic_num,
                    'predict_label': pred.item()
                })
    
    # 转换为DataFrame并排序
    df = pd.DataFrame(results)
    df = df.sort_values(by='pic_num').reset_index(drop=True)
    
    # 保存为Excel
    df.to_excel(CONFIG['save_path'], index=False)
    print(f"\n成功导出预测结果至: {CONFIG['save_path']}")
    print(f"共预测 {len(df)} 张图片")
    print("\n预测结果样例:")
    print(df.head(10))
    print("=" * 60)

# ================= 6. 验收测试接口 =================

def predict_new_images(test_image_dir, output_path='./test_result.xlsx'):
    """
    用于验收时测试新图片
    
    Args:
        test_image_dir: 测试图片目录
        output_path: 输出Excel路径
    """
    print("\n" + "=" * 60)
    print("验收模式: 测试新图片...")
    print("=" * 60)
    
    # 加载模型
    model_path = f"{CONFIG['model_save_dir']}/best_model.pth"
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return
    
    model = get_model().to(CONFIG['device'])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"已加载模型")
    
    # 获取测试图片
    test_imgs = [os.path.join(test_image_dir, i) 
                 for i in os.listdir(test_image_dir) 
                 if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"找到 {len(test_imgs)} 张测试图片")
    
    # 预测
    dataset = SceneDataset(test_imgs, mode='pred', transform=get_transforms('val'))
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
                
                results.append({
                    'pic_num': pic_num,
                    'predict_label': pred.item()
                })
    
    # 保存结果
    df = pd.DataFrame(results)
    df = df.sort_values(by='pic_num').reset_index(drop=True)
    df.to_xlsx(output_path, index=False)
    print(f"\n预测完成! 结果已保存至: {output_path}")
    print("=" * 60)

# ================= 7. 主函数 =================

if __name__ == "__main__":
    # 训练模型
    #train()
    
    # 预测pred_data
    predict()
    
    print("\n所有任务完成!")
    print("提示: 验收时使用 predict_new_images() 函数测试新图片")