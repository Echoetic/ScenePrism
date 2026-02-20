import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['figure.figsize'] = [12, 8]

# 数据准备
# 数据集性能
datasets = ['训练集', '验证集', '测试集', 'pred_data']
dataset_acc = [88.5, 86.2, 85.3, 85.0]
dataset_samples = [13600, 1700, 1700, 100]

# 各类别性能
categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
accuracy = [94.1, 93.8, 94.1, 76.5, 82.4, 68.8]
precision = [91.4, 93.8, 94.1, 81.3, 82.4, 73.3]
recall = [94.1, 93.8, 94.1, 76.5, 82.4, 68.8]
f1_scores = [0.927, 0.938, 0.941, 0.788, 0.824, 0.709]

# 平均性能
avg_metrics = [85.0, 86.0, 85.0, 0.854]

# 创建图形 - 主图
fig = plt.figure(figsize=(18, 12))
fig.suptitle('最终性能总结与各类别性能对比', fontsize=20, fontweight='bold', y=0.98)

# 子图1：数据集性能对比
ax1 = plt.subplot(2, 3, 1)
colors1 = ['#4C72B0', '#55A868', '#C44E52', '#FF6B6B']
bars1 = ax1.bar(datasets, dataset_acc, color=colors1, alpha=0.8)

# 添加数值标签
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 在条形图下方添加样本数量
    ax1.annotate(f'{dataset_samples[i]}张',
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, -25), textcoords="offset points",
                ha='center', va='top', fontsize=10)

# 突出显示目标达成
ax1.annotate('✓ 达成目标', 
            xy=(3, dataset_acc[3]), 
            xytext=(3.5, dataset_acc[3] + 2),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=12, color='green', fontweight='bold')

ax1.set_xlabel('数据集')
ax1.set_ylabel('准确率 (%)')
ax1.set_title('各数据集性能对比', fontsize=14, fontweight='bold')
ax1.set_ylim([80, 90])
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=85.0, color='red', linestyle='--', alpha=0.5, linewidth=1)

# 子图2：各类别准确率对比
ax2 = plt.subplot(2, 3, 2)
colors2 = plt.cm.Set3(np.linspace(0, 1, len(categories)))
bars2 = ax2.bar(categories, accuracy, color=colors2, alpha=0.8)

# 添加数值标签
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)
    
    # 添加排名标签
    rank = np.argsort(accuracy)[::-1][i] + 1
    ax2.annotate(f'#{rank}',
                xy=(bar.get_x() + bar.get_width() / 2, height/2),
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# 添加平均线
avg_acc = np.mean(accuracy)
ax2.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.text(len(categories)-0.5, avg_acc+1, f'平均: {avg_acc:.1f}%', 
         color='red', fontsize=11, fontweight='bold', ha='right')

ax2.set_xlabel('类别')
ax2.set_ylabel('准确率 (%)')
ax2.set_title('各类别准确率对比', fontsize=14, fontweight='bold')
# 修复：设置x轴刻度和标签
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels(categories, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# 子图3：各类别四个指标对比（分组条形图）
ax3 = plt.subplot(2, 3, 3)
x = np.arange(len(categories))
width = 0.2

bars_acc = ax3.bar(x - 1.5*width, accuracy, width, label='准确率', alpha=0.8)
bars_pre = ax3.bar(x - 0.5*width, precision, width, label='精确率', alpha=0.8)
bars_rec = ax3.bar(x + 0.5*width, recall, width, label='召回率', alpha=0.8)
bars_f1 = ax3.bar(x + 1.5*width, np.array(f1_scores)*100, width, label='F1分数(x100)', alpha=0.8)

# 添加数值标签（只显示前两个避免过于拥挤）
for i, bar in enumerate(bars_acc):
    if i % 2 == 0:  # 每隔一个显示
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

ax3.set_xlabel('类别')
ax3.set_ylabel('百分比 (%)')
ax3.set_title('各类别四个指标对比', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, rotation=45, ha='right')
ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
ax3.grid(True, alpha=0.3, axis='y')

# 子图4：各类别F1分数对比
ax4 = plt.subplot(2, 3, 4)
# 按F1分数排序
sorted_indices = np.argsort(f1_scores)[::-1]
sorted_categories = [categories[i] for i in sorted_indices]
sorted_f1 = [f1_scores[i] for i in sorted_indices]
colors_f1 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(categories)))

bars4 = ax4.barh(sorted_categories, sorted_f1, color=colors_f1, alpha=0.8)

# 添加数值标签
for i, bar in enumerate(bars4):
    width_val = bar.get_width()
    ax4.annotate(f'{width_val:.3f}',
                xy=(width_val, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 添加颜色强度指示
    color_intensity = (sorted_f1[i] - min(sorted_f1)) / (max(sorted_f1) - min(sorted_f1))
    ax4.annotate(f'★' * (int(color_intensity * 5) + 1),
                xy=(0, bar.get_y() + bar.get_height() / 2),
                xytext=(-25, 0), textcoords="offset points",
                ha='right', va='center', fontsize=12, color='gold')

ax4.set_xlabel('F1分数')
ax4.set_title('各类别F1分数对比（已排序）', fontsize=14, fontweight='bold')
ax4.set_xlim([0.65, 0.95])
ax4.axvline(x=np.mean(f1_scores), color='red', linestyle='--', linewidth=2, alpha=0.7)
ax4.text(np.mean(f1_scores)+0.01, 0.5, f'平均: {np.mean(f1_scores):.3f}', 
         transform=ax4.get_xaxis_transform(), color='red', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 子图5：各类别性能雷达图
ax5 = plt.subplot(2, 3, 5, projection='polar')

# 标准化数据用于雷达图
def normalize(data):
    return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))

# 雷达图指标
radar_categories = ['准确率', '精确率', '召回率', 'F1分数']
N = len(radar_categories)

# 为每个类别计算雷达图数据
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# 颜色映射
color_map = plt.cm.tab10(np.linspace(0, 1, len(categories)))

for i, category in enumerate(categories):
    # 收集四个指标
    values = [accuracy[i]/100, precision[i]/100, recall[i]/100, f1_scores[i]]
    values_normalized = normalize(values)  # 标准化到0-1范围
    
    # 调整数值范围，使图表更美观
    values_adjusted = values_normalized * 0.7 + 0.3  # 调整到0.3-1.0范围
    
    # 修复：正确闭合雷达图数据
    values_adjusted = np.concatenate([values_adjusted, [values_adjusted[0]]])  # 正确闭合
    
    ax5.plot(angles, values_adjusted, 'o-', linewidth=2, label=category, 
             color=color_map[i], alpha=0.7)
    ax5.fill(angles, values_adjusted, alpha=0.1, color=color_map[i])

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(radar_categories)
ax5.set_title('各类别四维性能雷达图', fontsize=14, fontweight='bold', pad=20)
ax5.set_ylim([0, 1.2])
ax5.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)

# 子图6：平均指标对比
ax6 = plt.subplot(2, 3, 6)
avg_metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
avg_metrics_values = [avg_metrics[0], avg_metrics[1], avg_metrics[2], avg_metrics[3]*100]  # F1分数乘以100

colors_avg = plt.cm.Paired(np.linspace(0, 1, len(avg_metrics_labels)))
bars6 = ax6.bar(avg_metrics_labels, avg_metrics_values, color=colors_avg, alpha=0.8)

# 添加数值标签
for i, bar in enumerate(bars6):
    height = bar.get_height()
    unit = '%' if i < 3 else ''
    value = f'{height:.1f}{unit}' if i < 3 else f'{height/100:.3f}'
    ax6.annotate(value,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加目标指示线
    if i == 0:  # 准确率
        ax6.axhline(y=85.0, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax6.text(len(avg_metrics_labels)-0.5, 85.0, '目标: 85.0%', 
                color='green', fontsize=10, fontweight='bold', ha='right')

ax6.set_xlabel('指标')
ax6.set_ylabel('值')
ax6.set_title('平均指标对比', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
plt.savefig('最终性能总结.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建第二个图表：详细分析图
fig2, ((ax7, ax8), (ax9, ax10)) = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('性能详细分析与难度评估', fontsize=18, fontweight='bold', y=0.98)

# 子图7：性能与难度关系散点图
# 假设难度评分（根据描述：glacier和forest最简单，street最难）
difficulty_scores = [3, 2, 1, 5, 4, 6]  # 1最简单，6最难

scatter = ax7.scatter(difficulty_scores, accuracy, s=np.array(f1_scores)*500, 
                      c=range(len(categories)), cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)

# 为每个点添加标签
for i, category in enumerate(categories):
    ax7.annotate(category, (difficulty_scores[i], accuracy[i]), 
                xytext=(5, 5), textcoords="offset points", fontsize=10)

# 添加趋势线
z = np.polyfit(difficulty_scores, accuracy, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(difficulty_scores), max(difficulty_scores), 100)
ax7.plot(x_trend, p(x_trend), 'r--', alpha=0.7, linewidth=2)

ax7.set_xlabel('难度评分 (1最简单, 6最难)')
ax7.set_ylabel('准确率 (%)')
ax7.set_title('类别准确率与难度关系', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('类别索引')

# 子图8：性能差异分析
performance_differences = np.abs(np.array(accuracy) - np.array(precision))
colors_diff = ['green' if diff <= 2 else 'orange' if diff <= 5 else 'red' for diff in performance_differences]

bars8 = ax8.bar(categories, performance_differences, color=colors_diff, alpha=0.7)

# 添加数值标签
for i, bar in enumerate(bars8):
    height = bar.get_height()
    ax8.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

ax8.set_xlabel('类别')
ax8.set_ylabel('准确率与精确率差异 (%)')
ax8.set_title('准确率与精确率差异分析', fontsize=14, fontweight='bold')
# 修复：设置x轴刻度和标签
ax8.set_xticks(range(len(categories)))
ax8.set_xticklabels(categories, rotation=45, ha='right')
ax8.grid(True, alpha=0.3, axis='y')

# 添加解释标签
ax8.text(0.02, 0.98, '绿色: ≤2% (优秀)\n橙色: 2-5% (良好)\n红色: >5% (需改进)', 
         transform=ax8.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 子图9：各类别性能热力图
metrics_matrix = np.array([accuracy, precision, recall, np.array(f1_scores)*100])
im = ax9.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=60, vmax=100)

# 设置坐标轴
ax9.set_xticks(np.arange(len(categories)))
ax9.set_yticks(np.arange(len(['准确率', '精确率', '召回率', 'F1分数(x100)'])))
ax9.set_xticklabels(categories)
ax9.set_yticklabels(['准确率', '精确率', '召回率', 'F1分数(x100)'])

# 添加数值标签
for i in range(len(['准确率', '精确率', '召回率', 'F1分数(x100)'])):
    for j in range(len(categories)):
        text = ax9.text(j, i, f'{metrics_matrix[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax9.set_title('各类别性能热力图', fontsize=14, fontweight='bold')

# 子图10：性能分布箱线图
data_for_boxplot = [accuracy, precision, recall, np.array(f1_scores)*100]
bp = ax10.boxplot(data_for_boxplot, labels=['准确率', '精确率', '召回率', 'F1分数(x100)'], 
                  patch_artist=True, showmeans=True)

# 设置箱线图颜色
colors_box = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 设置均值点样式
bp['means'][0].set(marker='o', markerfacecolor='blue', markersize=8)
bp['means'][1].set(marker='o', markerfacecolor='green', markersize=8)
bp['means'][2].set(marker='o', markerfacecolor='red', markersize=8)
bp['means'][3].set(marker='o', markerfacecolor='orange', markersize=8)

ax10.set_ylabel('百分比 (%)')
ax10.set_title('性能指标分布箱线图', fontsize=14, fontweight='bold')
ax10.grid(True, alpha=0.3, axis='y')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存第二个图片
plt.savefig('性能详细分析.png', dpi=300, bbox_inches='tight')
plt.show()

print("可视化已生成完成！")
print("1. 主图包含6个子图，展示了整体性能和各类别对比")
print("2. 详细分析图包含4个子图，深入分析了性能与难度的关系")
print("3. 图片已保存为'最终性能总结.png'和'性能详细分析.png'")