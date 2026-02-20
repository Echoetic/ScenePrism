import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import warnings
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 使用train_enhanced.py的配置
from enhanced_ensemble_85 import CONFIG, SceneDataset, get_transforms, get_model

class LRFinder:
    """学习率查找器 - 找到最优学习率"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.history = {'lr': [], 'loss': []}
        self.best_lr = None
    
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100, 
                   smooth_f=0.05, diverge_th=5):
        """
        执行学习率范围测试
        
        参数:
            train_loader: 训练数据加载器
            start_lr: 起始学习率
            end_lr: 结束学习率
            num_iter: 迭代次数
            smooth_f: 平滑因子
            diverge_th: 发散阈值
        """
        # 保存模型初始状态
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        
        # 初始化
        self.model.train()
        self.history = {'lr': [], 'loss': []}
        
        # 学习率调度器 - 指数增长
        lr_schedule = np.geomspace(start_lr, end_lr, num_iter)
        
        iterator = iter(train_loader)
        smoothed_loss = 0
        best_loss = float('inf')
        batch_num = 0
        
        print(f"{'='*70}")
        print("学习率查找器运行中...")
        print(f"范围: {start_lr:.2e} → {end_lr:.2e}")
        print(f"迭代次数: {num_iter}")
        print(f"{'='*70}\n")
        
        progress_bar = tqdm(range(num_iter), desc="LR查找")
        
        for iteration in progress_bar:
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)
            
            # 设置学习率
            lr = lr_schedule[iteration]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # 前向传播
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录
            self.history['lr'].append(lr)
            self.history['loss'].append(loss.item())
            
            # 平滑损失
            if iteration == 0:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = smooth_f * loss.item() + (1 - smooth_f) * smoothed_loss
            
            # 检查是否发散
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            if smoothed_loss > diverge_th * best_loss:
                print(f"\n\n训练发散，停止测试")
                break
            
            # 更新进度条
            progress_bar.set_postfix({
                'lr': f'{lr:.2e}',
                'loss': f'{smoothed_loss:.4f}'
            })
            
            batch_num += 1
        
        # 恢复模型状态
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        print(f"\n\n{'='*70}")
        print("学习率查找完成!")
        print(f"{'='*70}")
    
    def plot(self, skip_start=10, skip_end=5, log_lr=True, save_path='lr_finder.png'):
        """
        绘制学习率 vs 损失曲线
        
        参数:
            skip_start: 跳过开始的N个点
            skip_end: 跳过结束的N个点
            log_lr: 是否使用对数坐标
            save_path: 保存路径
        """
        if not self.history['lr']:
            print("错误: 需要先运行range_test()")
            return
        
        # 截取数据
        lrs = self.history['lr'][skip_start:-skip_end if skip_end > 0 else None]
        losses = self.history['loss'][skip_start:-skip_end if skip_end > 0 else None]
        
        # 找到最小损失对应的学习率
        min_loss_idx = losses.index(min(losses))
        min_loss_lr = lrs[min_loss_idx]
        
        # 找到梯度最大的学习率（推荐）
        # 通过计算损失的一阶导数
        grad = np.gradient(losses)
        max_grad_idx = np.argmin(grad)  # 最负的梯度
        suggested_lr = lrs[max_grad_idx]
        
        # 推荐学习率为最大梯度点的1/10
        self.best_lr = suggested_lr / 10
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 左图: LR vs Loss
        ax1.plot(lrs, losses, linewidth=2)
        ax1.axvline(min_loss_lr, color='red', linestyle='--', 
                   label=f'最小Loss LR: {min_loss_lr:.2e}', linewidth=1.5)
        ax1.axvline(suggested_lr, color='green', linestyle='--',
                   label=f'最大梯度 LR: {suggested_lr:.2e}', linewidth=1.5)
        ax1.axvline(self.best_lr, color='purple', linestyle='--',
                   label=f'推荐 LR: {self.best_lr:.2e}', linewidth=2)
        
        if log_lr:
            ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Learning Rate Finder', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 右图: Loss的变化率
        ax2.plot(lrs, grad, linewidth=2, color='orange')
        ax2.axvline(suggested_lr, color='green', linestyle='--',
                   label=f'最大梯度点: {suggested_lr:.2e}', linewidth=1.5)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        if log_lr:
            ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate', fontsize=12)
        ax2.set_ylabel('Loss变化率 (梯度)', fontsize=12)
        ax2.set_title('Loss Gradient', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}")
        plt.close()
        
        # 打印建议
        print(f"\n{'='*70}")
        print("📊 学习率分析结果")
        print(f"{'='*70}")
        print(f"最小损失对应的学习率: {min_loss_lr:.2e}")
        print(f"损失下降最快的学习率: {suggested_lr:.2e}")
        print(f"✅ 推荐学习率: {self.best_lr:.2e} (最大梯度的1/10)")
        print(f"{'='*70}")
        print("\n💡 使用建议:")
        print(f"1. 在CONFIG中设置: CONFIG['lr'] = {self.best_lr:.2e}")
        print(f"2. 如果想要更激进: CONFIG['lr'] = {suggested_lr:.2e}")
        print(f"3. 如果想要更保守: CONFIG['lr'] = {self.best_lr / 3:.2e}")
        print(f"{'='*70}\n")
        
        return self.best_lr

def find_optimal_lr(model_name='resnet50', num_samples=2000):
    """
    为指定模型找到最优学习率
    
    参数:
        model_name: 模型名称
        num_samples: 使用的样本数量
    """
    print(f"{'='*70}")
    print(f"为 {model_name} 查找最优学习率")
    print(f"{'='*70}\n")
    
    # 加载数据（使用部分数据加速）
    all_paths = []
    all_labels = []
    for cls_name, idx in CONFIG['class_map'].items():
        cls_folder = os.path.join(CONFIG['data_root'], cls_name)
        if os.path.exists(cls_folder):
            imgs = [os.path.join(cls_folder, i) for i in os.listdir(cls_folder) 
                   if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
            all_paths.extend(imgs)
            all_labels.extend([idx] * len(imgs))
    
    # 只使用部分数据
    if len(all_paths) > num_samples:
        indices = np.random.choice(len(all_paths), num_samples, replace=False)
        all_paths = [all_paths[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
    
    print(f"使用 {len(all_paths)} 张图片进行学习率搜索\n")
    
    # 创建数据集
    dataset = SceneDataset(all_paths, all_labels, 'train', 
                          get_transforms('train', 'medium'))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, 
                       num_workers=8, pin_memory=True)
    
    # 创建模型
    model = get_model(model_name).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-7, weight_decay=CONFIG['weight_decay'])
    
    # 运行学习率查找
    lr_finder = LRFinder(model, optimizer, criterion, CONFIG['device'])
    lr_finder.range_test(loader, start_lr=1e-7, end_lr=1, num_iter=100)
    
    # 绘制并获取推荐学习率
    best_lr = lr_finder.plot(save_path=f'lr_finder_{model_name}.png')
    
    return best_lr

def find_all_models_lr():
    """为所有模型找到最优学习率"""
    results = {}
    
    for model_name in CONFIG['models']:
        print(f"\n\n{'#'*70}")
        print(f"# 模型: {model_name}")
        print(f"{'#'*70}\n")
        
        try:
            best_lr = find_optimal_lr(model_name)
            results[model_name] = best_lr
        except Exception as e:
            print(f"错误: {e}")
            results[model_name] = None
    
    # 总结
    print(f"\n\n{'='*70}")
    print("📊 所有模型的推荐学习率汇总")
    print(f"{'='*70}")
    
    valid_lrs = [lr for lr in results.values() if lr is not None]
    if valid_lrs:
        avg_lr = np.mean(valid_lrs)
        
        for model_name, lr in results.items():
            if lr:
                print(f"{model_name:20s}: {lr:.2e}")
        
        print(f"\n{'='*70}")
        print(f"📌 平均推荐学习率: {avg_lr:.2e}")
        print(f"💡 建议在CONFIG中设置: CONFIG['lr'] = {avg_lr:.2e}")
        print(f"{'='*70}\n")
        
        return avg_lr
    else:
        print("错误: 没有成功的学习率搜索")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='学习率查找器')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='模型名称 (resnet50, efficientnet_b4, etc.)')
    parser.add_argument('--all', action='store_true',
                       help='为所有模型查找学习率')
    parser.add_argument('--samples', type=int, default=2000,
                       help='使用的样本数量')
    
    args = parser.parse_args()
    
    if args.all:
        find_all_models_lr()
    else:
        find_optimal_lr(args.model, args.samples)