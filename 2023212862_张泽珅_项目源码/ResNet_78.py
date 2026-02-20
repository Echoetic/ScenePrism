import os
import time
import copy
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    # 路径配置 (根据你的 tree 结构)
    DATA_ROOT = r"/root/autodl-tmp/Classify/data"
    TRAIN_DIR = os.path.join(DATA_ROOT, "basic_data")
    PRED_DIR = os.path.join(DATA_ROOT, "pred_data")
    OUTPUT_FILE = "pred_result.csv"
    MODEL_SAVE_PATH = "models/best_model.pth"
    
    # 硬件参数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8 # Ultra 9 核心多，可以适当调高，加速IO
    PIN_MEMORY = True
    
    # 训练超参数
    IMG_SIZE = 150
    BATCH_SIZE = 128  # 32GB 显存可以开得很大，128-256均可
    EPOCHS = 15       # 迁移学习通常不需要太久
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 6
    
    # 标签映射 (根据题目描述)
    CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    # 题目给定的标签映射: buildings0, forest1, ...
    LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

print(f"🚀 Running on device: {Config.DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. 数据处理与增强 (Data Processing)
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # 增加随机旋转，提升鲁棒性
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 稍微调整色彩
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 标准均值方差
    ]),
    'val': transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 预测集不需要增强，只需要归一化
    'pred': transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==========================================
# 3. 数据集准备 (Dataset Preparation)
# ==========================================

# 3.1 加载 Basic Data 并划分
full_dataset = datasets.ImageFolder(Config.TRAIN_DIR) # 原始数据集
# 确保 ImageFolder 读取的类别顺序与题目要求一致
# ImageFolder 默认按字母顺序排序 classes，我们需要核对一下
print(f"检测到的类别映射: {full_dataset.class_to_idx}")
# 如果 ImageFolder 的映射与题目要求的 0-5 不一致，需要手动调整，但此处按首字母排序恰好符合:
# buildings(0), forest(1), glacier(2), mountain(3), sea(4), street(5) -> 符合题目。

# 划分 训练集(80%) / 验证集(10%) / 测试集(10%)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42) # 固定随机种子以便复现
)

# 应用对应的 Transform (由于 random_split 只是子集引用，需要重写 Dataset 类或手动应用，
# 这里为了简便，我们在 Loader 阶段或者使用一个简单的 Wrapper)
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

train_set = TransformSubset(train_dataset, data_transforms['train'])
val_set = TransformSubset(val_dataset, data_transforms['val'])
test_set = TransformSubset(test_dataset, data_transforms['val'])

train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, 
                          num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=False, 
                        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False, 
                         num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

# 3.2 自定义预测数据集类 (针对 001.jpg 格式优化)
# ==========================================
class PredDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 获取所有图片文件
        all_files = glob.glob(os.path.join(root_dir, "*"))
        # 过滤非图片文件
        self.image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 排序逻辑优化：
        # 对于 001.jpg, 002.jpg 这种格式，直接使用字符串排序即可保证顺序正确
        # 但为了绝对稳健，我们依然提取数字部分进行排序
        self.image_files.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        
        # 打印前3个和最后3个文件，供你自查顺序是否正确
        print(f"Dataset Log: 检测到 {len(self.image_files)} 张预测图片")
        if len(self.image_files) > 0:
            print(f"Dataset Log: 排序首位文件: {os.path.basename(self.image_files[0])} -> ID: {int(re.search(r'\d+', os.path.basename(self.image_files[0])).group())}")
            print(f"Dataset Log: 排序末位文件: {os.path.basename(self.image_files[-1])} -> ID: {int(re.search(r'\d+', os.path.basename(self.image_files[-1])).group())}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 核心逻辑：提取文件名中的数字并转为整数
        # 例如: "001.jpg" -> re 提取出 "001" -> int("001") 变成 1
        # 这完美适配 csv 中的 pic_num 格式
        filename = os.path.basename(img_path)
        try:
            # 查找文件名中的第一个连续数字串
            pic_num_str = re.search(r'\d+', filename).group()
            pic_num = int(pic_num_str)
        except:
            # 万一文件名没有数字（极小概率），回退到使用索引
            print(f"Warning: 无法从文件名 {filename} 提取数字，使用索引代替")
            pic_num = idx + 1 
            
        if self.transform:
            image = self.transform(image)
            
        return image, pic_num

# 重新实例化 DataLoader
pred_dataset = PredDataset(Config.PRED_DIR, transform=data_transforms['pred'])
pred_loader = DataLoader(pred_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

# ==========================================
# 4. 模型构建 (Model Setup - Transfer Learning)
# ==========================================
def build_model():
    # 使用 ResNet50 预训练模型
    #  - 报告中可以插入 ResNet 结构图
    model = models.resnet50(pretrained=True)
    
    # 冻结所有层 (只训练全连接层) -> 也可以选择解冻最后几层 Fine-tune
    for param in model.parameters():
        param.requires_grad = False
        
    # 修改全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, Config.NUM_CLASSES)
    )
    
    return model.to(Config.DEVICE)

model = build_model()
criterion = nn.CrossEntropyLoss()
# 优化器只优化 fc 层的参数
optimizer = optim.AdamW(model.fc.parameters(), lr=Config.LEARNING_RATE)
# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# 混合精度 Scaler
scaler = GradScaler()

# ==========================================
# 5. 训练与验证流程 (Training Loop)
# ==========================================
def train_model(model, train_loader, val_loader, epochs):
    best_acc = 0.0
    
    # 确保 models 文件夹存在
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_set)
        epoch_acc = running_corrects.double() / len(train_set)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # --- 验证阶段 ---
        model.eval()
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_acc = val_running_corrects.double() / len(val_set)
        print(f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print("✨ Best model saved!")
            
        scheduler.step()

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")
    return model

# ==========================================
# 6. 执行训练
# ==========================================
# 如果只是测试，可以注释掉下面这行，加载已保存的模型
print("开始训练模型...")
model = train_model(model, train_loader, val_loader, Config.EPOCHS)

# 加载最佳模型参数
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

# ==========================================
# 7. 测试集评估 (Evaluation on Test Set)
# ==========================================
print("\n在独立测试集上评估...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 输出分类报告 (Precision, Recall, F1-Score)
print(classification_report(all_labels, all_preds, target_names=Config.CLASS_NAMES))

# ==========================================
# 8. 预测并生成结果文件 (Prediction & Export)
# ==========================================
print(f"\n正在对 {len(pred_dataset)} 张图片进行预测...")
model.eval()
results = [] # 存储结果 [pic_num, predict_label]

with torch.no_grad():
    for inputs, pic_nums in tqdm(pred_loader, desc="Predicting"):
        inputs = inputs.to(Config.DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        # 将 tensor 转为 list
        preds = preds.cpu().numpy()
        pic_nums = pic_nums.numpy()
        
        for num, label in zip(pic_nums, preds):
            results.append({'pic_num': num, 'predict_label': label})

# 创建 DataFrame 并保存
df = pd.DataFrame(results)

# 确保按照 pic_num 排序 (如果之前是乱序的)
df = df.sort_values(by='pic_num')

# 检查是否有附件中要求的列名
print("预览前5行数据:")
print(df.head())

# 保存为 CSV (注意：题目虽然说 .xlsx, 但附件和输出示例通常用 csv 更稳妥，
# 如果严格要求 xlsx，请将下面的 to_csv 改为 to_excel，并安装 openpyxl)
# 根据你提供的附件是 csv，这里优先生成 csv
output_csv_path = os.path.join(os.path.dirname(Config.DATA_ROOT), Config.OUTPUT_FILE)
df.to_csv(output_csv_path, index=False)
print(f"✅ 结果已保存至: {output_csv_path}")