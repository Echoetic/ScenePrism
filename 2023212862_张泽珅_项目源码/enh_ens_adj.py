import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    
    # ===== 关键超参数 - 可调整 =====
    'batch_size': 96,           # 可选: 64, 80, 96, 112, 128
    'lr': 0.0005,               # 可选: 0.0001, 0.0003, 0.0005, 0.001
    'weight_decay': 0.0001,     # 可选: 0.00001, 0.0001, 0.001
    'epochs': 60,               # 可选: 50, 60, 80, 100
    'warmup_epochs': 5,         # 学习率预热轮数
    
    # ===== 数据增强强度 =====
    'aug_strength': 'strong',   # 可选: 'light', 'medium', 'strong', 'very_strong'
    'mixup_alpha': 0.2,         # Mixup增强强度，0表示不使用
    'cutmix_alpha': 1.0,        # CutMix增强强度，0表示不使用
    'cutmix_prob': 0.5,         # CutMix使用概率
    
    # ===== 模型配置 =====
    'use_ema': True,            # 使用指数移动平均
    'ema_decay': 0.9995,
    'label_smoothing': 0.1,     # 标签平滑
    'dropout': 0.5,             # Dropout比例
    
    # ===== 训练策略 =====
    'use_sam': False,           # 使用SAM优化器（更慢但可能更好）
    'gradient_clip': 1.0,       # 梯度裁剪
    'early_stop_patience': 15,  # 早停耐心值
    'use_swa': True,            # 使用随机权重平均
    'swa_start': 40,            # SWA开始的epoch
    
    # ===== 测试时增强 =====
    'use_tta': True,            # 测试时增强
    'tta_transforms': 5,        # TTA的增强次数
    
    # ===== 模型集成 =====
    'models': ['resnet50', 'resnet101', 'efficientnet_b4', 'convnext_small'],
    
    # 固定参数
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
print(f"改进的集成模型 - 目标准确率 >85%")
print(f"{'='*70}")
print(f"设备: {CONFIG['device']}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"{'='*70}")

# ================= 数据集和增强 =================

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

def get_transforms(mode='train', strength='medium'):
    """
    获取数据增强变换
    strength: 'light', 'medium', 'strong', 'very_strong'
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        if strength == 'light':
            return transforms.Compose([
                transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif strength == 'medium':
            return transforms.Compose([
                transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))
            ])
        elif strength == 'strong':
            return transforms.Compose([
                transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.15))
            ])
        else:  # very_strong
            return transforms.Compose([
                transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
            ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """生成随机裁剪框"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# ================= 模型定义 =================

class ModelEMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.9999):
        self.module = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def update(self):
        with torch.no_grad():
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def register(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def get_model(model_name='resnet50'):
    """获取模型"""
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout'] * 0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout'] * 0.3),
            nn.Linear(512, CONFIG['num_classes'])
        )
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout'] * 0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout'] * 0.3),
            nn.Linear(512, CONFIG['num_classes'])
        )
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout'] * 0.3),
            nn.Linear(512, CONFIG['num_classes'])
        )
    elif model_name == 'convnext_small':
        model = models.convnext_small(pretrained=True)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, CONFIG['num_classes'])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# ================= 训练函数 =================

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization优化器"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure"
        closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

def train_single_model(model_name, train_loader, val_loader, fold=0):
    """训练单个模型"""
    print(f"\n{'='*70}")
    print(f"训练模型: {model_name} (Fold {fold})")
    print(f"{'='*70}")
    
    model = get_model(model_name).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    
    # 选择优化器
    if CONFIG['use_sam']:
        base_optimizer = optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, 
                       lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], 
                               weight_decay=CONFIG['weight_decay'])
    
    # 学习率调度器 - 使用CosineAnnealing + Warmup
    def warmup_cosine_schedule(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return (epoch + 1) / CONFIG['warmup_epochs']
        else:
            progress = (epoch - CONFIG['warmup_epochs']) / (CONFIG['epochs'] - CONFIG['warmup_epochs'])
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    # EMA
    if CONFIG['use_ema']:
        ema = ModelEMA(model, decay=CONFIG['ema_decay'])
        ema.register()
    
    # SWA
    if CONFIG['use_swa']:
        swa_model = optim.swa_utils.AveragedModel(model)
        swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=CONFIG['lr'] * 0.1)
    
    scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] else None
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["epochs"]} [训练]', leave=False)
        for images, labels in train_bar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            # Mixup/CutMix增强
            r = np.random.rand(1)
            if CONFIG['mixup_alpha'] > 0 and r < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, CONFIG['mixup_alpha'])
                mixed = True
            elif CONFIG['cutmix_alpha'] > 0 and r < CONFIG['cutmix_prob']:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, CONFIG['cutmix_alpha'])
                mixed = True
            else:
                mixed = False
            
            if CONFIG['use_sam']:
                # SAM优化器需要两次前向传播
                def closure():
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=CONFIG['mixed_precision']):
                        outputs = model(images)
                        if mixed:
                            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                        else:
                            loss = criterion(outputs, labels)
                    if CONFIG['mixed_precision']:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    return loss
                
                loss = optimizer.step(closure)
                if CONFIG['mixed_precision']:
                    scaler.update()
            else:
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=CONFIG['mixed_precision']):
                    outputs = model(images)
                    if mixed:
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    else:
                        loss = criterion(outputs, labels)
                
                if CONFIG['mixed_precision']:
                    scaler.scale(loss).backward()
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
                    optimizer.step()
            
            # 更新EMA
            if CONFIG['use_ema']:
                ema.update()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            if mixed:
                train_correct += (lam * predicted.eq(labels_a).sum().item() + 
                                (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # 更新学习率
        if CONFIG['use_swa'] and epoch >= CONFIG['swa_start']:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # 验证阶段
        if CONFIG['use_ema']:
            ema.apply_shadow()
        
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
        
        if CONFIG['use_ema']:
            ema.restore()
        
        print(f"Epoch {epoch+1:3d} | 训练 Acc: {train_acc:5.2f}% | 验证 Acc: {val_acc:5.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存最佳模型
            save_dict = {
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }
            if CONFIG['use_ema']:
                save_dict['ema_state_dict'] = ema.shadow
            
            torch.save(save_dict, f"{CONFIG['model_save_dir']}/best_{model_name}_fold{fold}.pth")
            print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"  早停: 验证准确率已连续 {CONFIG['early_stop_patience']} 轮未提升")
                break
    
    # 如果使用SWA，保存SWA模型
    if CONFIG['use_swa']:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=torch.device("cuda"))
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'val_acc': best_val_acc
        }, f"{CONFIG['model_save_dir']}/swa_{model_name}_fold{fold}.pth")
    
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
    train_dataset = SceneDataset(X_train, y_train, 'train', 
                                get_transforms('train', CONFIG['aug_strength']))
    val_dataset = SceneDataset(X_val, y_val, 'val', get_transforms('val'))
    test_dataset = SceneDataset(X_test, y_test, 'test', get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                             num_workers=CONFIG['num_workers'], pin_memory=True, 
                             persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                           num_workers=CONFIG['num_workers'], pin_memory=True, 
                           persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    
    # 训练所有模型
    results = {}
    for model_name in CONFIG['models']:
        best_acc = train_single_model(model_name, train_loader, val_loader, fold=0)
        results[model_name] = best_acc
    
    # 测试集评估
    print(f"\n{'='*70}")
    print("测试集评估")
    print(f"{'='*70}")
    
    for model_name in CONFIG['models']:
        model = get_model(model_name).to(CONFIG['device'])
        
        # 尝试加载SWA模型，如果没有则加载普通模型
        swa_path = f"{CONFIG['model_save_dir']}/swa_{model_name}_fold0.pth"
        normal_path = f"{CONFIG['model_save_dir']}/best_{model_name}_fold0.pth"
        
        if CONFIG['use_swa'] and os.path.exists(swa_path):
            checkpoint = torch.load(swa_path)
            print(f"加载SWA模型: {model_name}")
        else:
            checkpoint = torch.load(normal_path)
        
        model.load_state_dict(checkpoint['model_state_dict'],weights_only=False)
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
        print(f"{model_name:20s} | 验证: {results[model_name]:5.2f}% | 测试: {test_acc:5.2f}%")
    
    print(f"{'='*70}")

# ================= 预测函数（带TTA） =================

def predict():
    """集成预测（带测试时增强）"""
    print(f"\n{'='*70}")
    print("模型预测")
    print(f"{'='*70}")
    
    # 加载所有模型
    models_list = []
    for model_name in CONFIG['models']:
        model = get_model(model_name).to(CONFIG['device'])
        
        # 优先使用SWA模型
        swa_path = f"{CONFIG['model_save_dir']}/swa_{model_name}_fold0.pth"
        normal_path = f"{CONFIG['model_save_dir']}/best_{model_name}_fold0.pth"
        
        if CONFIG['use_swa'] and os.path.exists(swa_path):
            checkpoint = torch.load(swa_path)
            model_type = "SWA"
        elif os.path.exists(normal_path):
            checkpoint = torch.load(normal_path)
            model_type = "Best"
        else:
            print(f"警告: 未找到模型 {model_name}")
            continue
        
        model.load_state_dict(checkpoint['model_state_dict'],weights_only=False)
        model.eval()
        models_list.append(model)
        print(f"  ✓ {model_name} ({model_type}) - 验证准确率: {checkpoint.get('val_acc', 0):.2f}%")
    
    if not models_list:
        print("错误: 没有可用的模型")
        return
    
    # 获取预测图片
    pred_imgs = [os.path.join(CONFIG['pred_root'], i) 
                 for i in os.listdir(CONFIG['pred_root']) 
                 if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"\n待预测图片: {len(pred_imgs)} 张")
    if CONFIG['use_tta']:
        print(f"使用TTA: {CONFIG['tta_transforms']} 次增强")
    
    # TTA变换
    tta_transforms = []
    if CONFIG['use_tta']:
        base_transform = get_transforms('val')
        tta_transforms = [
            base_transform,  # 原始
            transforms.Compose([transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
                              transforms.RandomHorizontalFlip(p=1.0),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            transforms.Compose([transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
                              transforms.RandomVerticalFlip(p=1.0),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            transforms.Compose([transforms.Resize((int(CONFIG['image_size'] * 1.1), int(CONFIG['image_size'] * 1.1))),
                              transforms.CenterCrop(CONFIG['image_size']),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            transforms.Compose([transforms.Resize((int(CONFIG['image_size'] * 1.2), int(CONFIG['image_size'] * 1.2))),
                              transforms.CenterCrop(CONFIG['image_size']),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        ]
    else:
        tta_transforms = [get_transforms('val')]
    
    results = []
    
    with torch.no_grad():
        for img_path in tqdm(pred_imgs, desc='预测进度'):
            try:
                fname = os.path.basename(img_path)
                image_pil = Image.open(img_path).convert('RGB')
                
                all_probs = []
                
                # 对每个TTA变换
                for transform in tta_transforms:
                    image = transform(image_pil).unsqueeze(0).to(CONFIG['device'])
                    
                    # 对每个模型
                    for model in models_list:
                        outputs = model(image)
                        probs = torch.softmax(outputs, dim=1)
                        all_probs.append(probs.cpu().numpy())
                
                # 平均所有预测概率
                avg_probs = np.mean(all_probs, axis=0)
                pred = np.argmax(avg_probs)
                
                try:
                    pic_num = int(os.path.splitext(fname)[0])
                except ValueError:
                    pic_num = fname
                
                results.append({'pic_num': pic_num, 'predict_label': int(pred)})
                
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {e}")
                continue
    
    # 保存结果
    df = pd.DataFrame(results)
    df = df.sort_values(by='pic_num').reset_index(drop=True)
    df.to_excel(CONFIG['save_path'], index=False)
    
    print(f"\n✓ 预测完成! 结果已保存至: {CONFIG['save_path']}")
    print(f"共预测 {len(df)} 张图片")
    print(f"{'='*70}")

def predict_new_images(test_image_dir, output_path='./test_result.xlsx'):
    """验收测试接口"""
    CONFIG['pred_root'] = test_image_dir
    CONFIG['save_path'] = output_path
    predict()

if __name__ == "__main__":
    train()
    predict()