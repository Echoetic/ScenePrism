import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['figure.figsize'] = [12, 8]

# 从训练日志中提取的数据
epochs = list(range(1, 61))
train_acc = [
    57.45, 80.35, 85.32, 86.28, 88.69, 89.57, 89.95, 90.46, 91.02, 92.01,
    92.10, 91.33, 91.96, 92.31, 90.86, 92.57, 91.54, 92.99, 92.76, 94.71,
    95.39, 92.62, 93.14, 92.56, 92.97, 94.74, 91.59, 93.61, 92.91, 93.83,
    93.79, 94.36, 93.98, 94.49, 93.06, 94.42, 93.87, 92.98, 94.72, 95.02,
    95.14, 94.79, 95.19, 95.38, 93.38, 95.51, 95.99, 94.54, 96.27, 94.31,
    93.60, 93.17, 93.30, 94.36, 94.21, 96.26, 95.62, 95.64, 94.53, 94.46
]
val_acc = [
    20.71, 42.35, 67.29, 79.71, 85.76, 88.29, 89.53, 91.00, 91.18, 92.24,
    92.59, 92.76, 93.12, 93.18, 93.76, 93.53, 93.71, 93.82, 94.00, 94.00,
    94.59, 94.00, 94.29, 94.06, 94.00, 94.29, 94.18, 94.65, 94.65, 94.88,
    94.71, 94.59, 94.53, 94.82, 94.82, 95.00, 94.88, 95.06, 95.12, 95.00,
    95.06, 95.18, 94.94, 95.12, 95.24, 95.18, 95.12, 95.18, 95.24, 95.35,
    95.35, 95.41, 95.24, 95.35, 95.35, 95.29, 95.35, 95.24, 95.41, 95.53
]
learning_rates = [
    0.000097, 0.000145, 0.000194, 0.000242, 0.000242, 0.000242, 0.000241, 0.000240, 0.000239, 0.000237,
    0.000235, 0.000232, 0.000230, 0.000226, 0.000223, 0.000219, 0.000215, 0.000210, 0.000205, 0.000200,
    0.000195, 0.000189, 0.000183, 0.000177, 0.000171, 0.000165, 0.000158, 0.000152, 0.000145, 0.000138,
    0.000131, 0.000124, 0.000118, 0.000111, 0.000104, 0.000097, 0.000090, 0.000084, 0.000077, 0.000071,
    0.000070, 0.000066, 0.000061, 0.000055, 0.000047, 0.000040, 0.000034, 0.000029, 0.000025, 0.000024,
    0.000024, 0.000024, 0.000024, 0.000024, 0.000024, 0.000024, 0.000024, 0.000024, 0.000024, 0.000024
]

# 标记保存最佳模型的epoch（验证准确率提升的时刻）
best_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 28, 30, 36, 38, 39, 42, 45, 50, 52, 60]
best_val_acc = [val_acc[i-1] for i in best_epochs]  # 对应epoch的验证准确率

# 创建图形
fig = plt.figure(figsize=(14, 10))
fig.suptitle('EfficientNet-B4 训练过程可视化 (Fold 0)', fontsize=18, fontweight='bold', y=0.98)

# 子图1：训练和验证准确率曲线
ax1 = plt.subplot(2, 1, 1)

# 绘制训练和验证准确率曲线
line_train, = ax1.plot(epochs, train_acc, 'b-', linewidth=2, label='训练准确率', alpha=0.8, marker='o', markersize=4)
line_val, = ax1.plot(epochs, val_acc, 'r-', linewidth=2, label='验证准确率', alpha=0.8, marker='s', markersize=4)

# 标记最佳模型保存点
scatter_best = ax1.scatter(best_epochs, best_val_acc, c='green', s=100, marker='*', edgecolors='black', linewidth=1, 
                          label='保存最佳模型', zorder=5)

# 添加最佳模型标签
for i, epoch in enumerate(best_epochs):
    if epoch in [1, 15, 30, 45, 60]:  # 只在关键epoch添加标签，避免过于拥挤
        ax1.annotate(f'最佳{val_acc[epoch-1]:.1f}%',
                    xy=(epoch, val_acc[epoch-1]),
                    xytext=(0, 10), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# 标记最终最佳验证准确率
final_best_val = max(val_acc)
final_best_epoch = val_acc.index(final_best_val) + 1
ax1.annotate(f'最终最佳: {final_best_val:.1f}% (Epoch {final_best_epoch})',
            xy=(final_best_epoch, final_best_val),
            xytext=(10, 10), textcoords="offset points",
            ha='left', va='bottom', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

# 计算并标记过拟合区域（训练准确率远高于验证准确率的区域）
overfit_threshold = 5  # 定义过拟合阈值为5%
overfit_epochs = []
for i in range(len(epochs)):
    if train_acc[i] - val_acc[i] > overfit_threshold:
        overfit_epochs.append(epochs[i])

if overfit_epochs:
    # 标记第一个过拟合点
    first_overfit = overfit_epochs[0]
    idx = first_overfit - 1
    ax1.annotate(f'可能过拟合\n差距: {train_acc[idx]-val_acc[idx]:.1f}%',
                xy=(first_overfit, train_acc[idx]),
                xytext=(15, 15), textcoords="offset points",
                ha='left', va='bottom', fontsize=9,
                arrowprops=dict(arrowstyle="->", color='orange', lw=1),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))
    
    # 用浅色区域标记所有过拟合点
    for epoch in overfit_epochs:
        idx = epoch - 1
        ax1.fill_between([epoch-0.5, epoch+0.5], 
                         val_acc[idx], train_acc[idx],
                         color='orange', alpha=0.1)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('准确率 (%)')
ax1.set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
ax1.set_xlim([0, 61])
ax1.set_ylim([0, 100])
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')

# 添加训练阶段标注
ax1.axvspan(1, 15, alpha=0.1, color='green', label='快速提升期')
ax1.axvspan(16, 30, alpha=0.1, color='yellow', label='稳定提升期')
ax1.axvspan(31, 60, alpha=0.1, color='red', label='微调期')

# 添加阶段文本标注
ax1.text(8, 20, '快速提升期', ha='center', va='center', fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
ax1.text(23, 20, '稳定提升期', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
ax1.text(45, 20, '微调期', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))

# 子图2：学习率变化曲线
ax2 = plt.subplot(2, 1, 2)

# 绘制学习率曲线
line_lr, = ax2.plot(epochs, learning_rates, 'g-', linewidth=2, label='学习率', alpha=0.8, marker='^', markersize=4)

# 标记学习率变化阶段
# 找到学习率变化的关键点
lr_changes = []
for i in range(1, len(learning_rates)):
    if learning_rates[i] != learning_rates[i-1]:
        lr_changes.append(i+1)  # i+1对应epoch编号

# 标记学习率变化点
for change_epoch in lr_changes:
    if change_epoch in [1, 5, 20, 30, 36, 45, 50]:  # 只标记关键变化点
        ax2.annotate(f'LR: {learning_rates[change_epoch-1]:.6f}',
                    xy=(change_epoch, learning_rates[change_epoch-1]),
                    xytext=(0, 10), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.5))

# 标记学习率调度策略
ax2.text(10, 0.00022, 'Warmup阶段\n(学习率递增)', ha='center', va='center', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
ax2.text(25, 0.00012, 'Cosine退火\n(学习率递减)', ha='center', va='center', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
ax2.text(55, 0.000024, '稳定学习率\n(微调阶段)', ha='center', va='center', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))

ax2.set_xlabel('Epoch')
ax2.set_ylabel('学习率')
ax2.set_title('学习率调度曲线', fontsize=14, fontweight='bold')
ax2.set_xlim([0, 61])
ax2.set_ylim([0, max(learning_rates) * 1.1])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
plt.savefig('EfficientNet_B4_训练过程.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建第二个图表：详细统计分析
fig2, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('EfficientNet-B4 训练过程详细分析', fontsize=18, fontweight='bold', y=0.98)

# 子图3：准确率提升速度分析
ax3.plot(epochs, train_acc, 'b-', linewidth=2, label='训练准确率', alpha=0.7)
ax3.plot(epochs, val_acc, 'r-', linewidth=2, label='验证准确率', alpha=0.7)

# 计算并绘制移动平均
window_size = 5
train_acc_ma = np.convolve(train_acc, np.ones(window_size)/window_size, mode='valid')
val_acc_ma = np.convolve(val_acc, np.ones(window_size)/window_size, mode='valid')
epochs_ma = epochs[window_size-1:]

ax3.plot(epochs_ma, train_acc_ma, 'b--', linewidth=1.5, label=f'训练准确率({window_size}epoch移动平均)', alpha=0.9)
ax3.plot(epochs_ma, val_acc_ma, 'r--', linewidth=1.5, label=f'验证准确率({window_size}epoch移动平均)', alpha=0.9)

# 标记收敛点（当验证准确率变化小于0.1%持续5个epoch时）
convergence_threshold = 0.1
convergence_epoch = None
for i in range(len(val_acc)-5):
    if max(val_acc[i:i+5]) - min(val_acc[i:i+5]) < convergence_threshold:
        convergence_epoch = i + 1
        break

if convergence_epoch:
    ax3.axvline(x=convergence_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.text(convergence_epoch, 50, f'收敛点: Epoch {convergence_epoch}', 
             rotation=90, ha='right', va='bottom', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

ax3.set_xlabel('Epoch')
ax3.set_ylabel('准确率 (%)')
ax3.set_title('准确率曲线与移动平均', fontsize=14, fontweight='bold')
ax3.set_xlim([0, 61])
ax3.set_ylim([0, 100])
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right')

# 子图4：训练与验证准确率差距分析
accuracy_gap = [train_acc[i] - val_acc[i] for i in range(len(epochs))]

bars = ax4.bar(epochs, accuracy_gap, color='orange', alpha=0.7, edgecolor='darkorange', linewidth=0.5)

# 标记过拟合区域
overfit_epochs_indices = [i for i, gap in enumerate(accuracy_gap) if gap > overfit_threshold]
for idx in overfit_epochs_indices:
    bars[idx].set_color('red')
    bars[idx].set_alpha(0.9)

# 添加平均差距线
avg_gap = np.mean(accuracy_gap)
ax4.axhline(y=avg_gap, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax4.text(60, avg_gap+0.5, f'平均差距: {avg_gap:.2f}%', 
         ha='right', va='bottom', fontsize=10, fontweight='bold')

# 添加过拟合阈值线
ax4.axhline(y=overfit_threshold, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax4.text(60, overfit_threshold+0.5, f'过拟合阈值: {overfit_threshold}%', 
         ha='right', va='bottom', fontsize=9, color='red')

ax4.set_xlabel('Epoch')
ax4.set_ylabel('训练-验证准确率差距 (%)')
ax4.set_title('过拟合分析: 训练与验证准确率差距', fontsize=14, fontweight='bold')
ax4.set_xlim([0, 61])
ax4.set_ylim([min(accuracy_gap)-1, max(accuracy_gap)+1])
ax4.grid(True, alpha=0.3, axis='y')

# 添加图例说明
ax4.text(0.02, 0.98, '红色: 可能过拟合\n(差距>5%)', 
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 子图5：学习率与验证准确率关系
ax5_lr = ax5
line_lr2, = ax5_lr.plot(epochs, learning_rates, 'g-', linewidth=2, label='学习率', alpha=0.8)
ax5_lr.set_xlabel('Epoch')
ax5_lr.set_ylabel('学习率', color='green')
ax5_lr.tick_params(axis='y', labelcolor='green')
ax5_lr.set_ylim([0, max(learning_rates)*1.1])

# 创建第二个y轴用于验证准确率
ax5_acc = ax5_lr.twinx()
line_val2, = ax5_acc.plot(epochs, val_acc, 'r-', linewidth=2, label='验证准确率', alpha=0.8)
ax5_acc.set_ylabel('验证准确率 (%)', color='red')
ax5_acc.tick_params(axis='y', labelcolor='red')
ax5_acc.set_ylim([0, 100])

# 标记学习率变化对准确率的影响
for change_epoch in lr_changes:
    if change_epoch < len(epochs):
        # 计算学习率变化后验证准确率的变化
        if change_epoch < len(val_acc):
            prev_acc = val_acc[change_epoch-2] if change_epoch > 1 else val_acc[0]
            curr_acc = val_acc[change_epoch-1]
            acc_change = curr_acc - prev_acc
            
            # 只在显著变化时标记
            if abs(acc_change) > 0.5:
                color = 'green' if acc_change > 0 else 'red'
                ax5.annotate(f'{acc_change:+.1f}%',
                           xy=(change_epoch, val_acc[change_epoch-1]),
                           xytext=(0, 15 if acc_change > 0 else -15), 
                           textcoords="offset points",
                           ha='center', va='bottom' if acc_change > 0 else 'top',
                           fontsize=8, color=color, fontweight='bold',
                           arrowprops=dict(arrowstyle="->", color=color, lw=1))

ax5.set_title('学习率与验证准确率关系', fontsize=14, fontweight='bold')
ax5_lr.set_xlim([0, 61])

# 合并图例
lines = [line_lr2, line_val2]
labels = [line.get_label() for line in lines]
ax5_lr.legend(lines, labels, loc='upper left')

# 子图6：训练过程统计摘要
ax6.axis('off')  # 关闭坐标轴

# 计算关键统计数据
max_train_acc = max(train_acc)
max_val_acc = max(val_acc)
final_train_acc = train_acc[-1]
final_val_acc = val_acc[-1]

train_improvement = max_train_acc - train_acc[0]
val_improvement = max_val_acc - val_acc[0]

avg_train_acc = np.mean(train_acc)
avg_val_acc = np.mean(val_acc)

# 计算训练速度（前10个epoch的平均提升）
early_train_speed = (train_acc[9] - train_acc[0]) / 9 if len(train_acc) > 9 else 0
early_val_speed = (val_acc[9] - val_acc[0]) / 9 if len(val_acc) > 9 else 0

# 创建统计摘要文本
summary_text = f"""
训练过程统计摘要 (EfficientNet-B4, Fold 0)

总体表现:
• 最终训练准确率: {final_train_acc:.2f}%
• 最终验证准确率: {final_val_acc:.2f}%
• 最佳验证准确率: {max_val_acc:.2f}% (Epoch {final_best_epoch})

提升幅度:
• 训练准确率提升: {train_improvement:.2f}% (从 {train_acc[0]:.2f}% 到 {max_train_acc:.2f}%)
• 验证准确率提升: {val_improvement:.2f}% (从 {val_acc[0]:.2f}% 到 {max_val_acc:.2f}%)

平均表现:
• 平均训练准确率: {avg_train_acc:.2f}%
• 平均验证准确率: {avg_val_acc:.2f}%
• 平均准确率差距: {avg_gap:.2f}%

训练速度:
• 早期训练速度: {early_train_speed:.2f}%/epoch (前10个epoch)
• 早期验证速度: {early_val_speed:.2f}%/epoch (前10个epoch)

收敛分析:
• 收敛点: Epoch {convergence_epoch if convergence_epoch else "未明显收敛"}
• 最佳模型保存次数: {len(best_epochs)}次
• 过拟合epoch数量: {len(overfit_epochs)}个

学习率调度:
• 初始学习率: {learning_rates[0]:.6f}
• 最终学习率: {learning_rates[-1]:.6f}
• 学习率变化次数: {len(lr_changes)}次
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax6.set_title('训练过程统计摘要', fontsize=14, fontweight='bold')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存第二个图片
plt.savefig('EfficientNet_B4_训练分析.png', dpi=300, bbox_inches='tight')
plt.show()

print("可视化已生成完成！")
print("1. 主图包含训练/验证准确率曲线和学习率曲线")
print("2. 详细分析图包含准确率分析、过拟合分析、学习率关系分析和统计摘要")
print("3. 图片已保存为'EfficientNet_B4_训练过程.png'和'EfficientNet_B4_训练分析.png'")