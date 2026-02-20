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
    'batch_size': 64,           # ViT需要较大显存
    'lr': 0.0003,               # ViT通常用较小的学习率
    'epochs': 50,
    'image_size': 224,
    'patch_size': 16,           # 将图像分成16x16的patch
    'num_classes': 6,
    'num_workers': 16,
    'seed': 42,
    'class_map': {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5},
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,
    
    # ViT特定参数
    'embed_dim': 768,           # embedding维度
    'depth': 12,                # Transformer层数
    'num_heads': 12,            # 注意力头数
    'mlp_ratio': 4.0,           # MLP隐藏层倍数
    'dropout': 0.1,
    'attention_dropout': 0.1,
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
print(f"Vision Transformer 场景分类模型")
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
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

# ================= Vision Transformer 模型 =================

class PatchEmbed(nn.Module):
    """将图像分割成patches并进行embedding"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """前馈神经网络"""
    def __init__(self, in_features, hidden_features, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0., attn_dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout)
    
    def forward(self, x):
        # 注意力 + 残差连接
        x = x + self.attn(self.norm1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer模型"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=6,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 dropout=0.1, attn_dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        
        # Normalization and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # 添加class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 提取class token并分类
        x = self.norm(x)
        x = x[:, 0]  # 只取class token
        x = self.head(x)
        
        return x

def get_model():
    """创建ViT模型"""
    model = VisionTransformer(
        img_size=CONFIG['image_size'],
        patch_size=CONFIG['patch_size'],
        num_classes=CONFIG['num_classes'],
        embed_dim=CONFIG['embed_dim'],
        depth=CONFIG['depth'],
        num_heads=CONFIG['num_heads'],
        mlp_ratio=CONFIG['mlp_ratio'],
        dropout=CONFIG['dropout'],
        attn_dropout=CONFIG['attention_dropout']
    )
    return model

# ================= 训练函数 =================

def train():
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
    print("构建Vision Transformer模型")
    print(f"{'='*70}")
    
    model = get_model().to(CONFIG['device'])
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器和学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
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
        scheduler.step()
        
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
            }, f"{CONFIG['model_save_dir']}/best_vit.pth")
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
    
    checkpoint = torch.load(f"{CONFIG['model_save_dir']}/best_vit.pth")
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

def predict():
    print(f"\n{'='*70}")
    print("模型预测")
    print(f"{'='*70}")
    
    model_path = f"{CONFIG['model_save_dir']}/best_vit.pth"
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return
    
    model = get_model().to(CONFIG['device'])
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

def predict_new_images(test_image_dir, output_path='./test_result.xlsx'):
    """验收测试接口"""
    CONFIG['pred_root'] = test_image_dir
    CONFIG['save_path'] = output_path
    predict()

if __name__ == "__main__":
    train()
    predict()