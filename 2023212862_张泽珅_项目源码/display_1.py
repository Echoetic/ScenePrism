import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['figure.figsize'] = [12, 8]

# 数据准备 - 转换为numpy数组
models = ['ResNet50', 'ResNet101', 'EfficientNet-B4', 'ConvNeXt-Small']
parameters = np.array([25, 44, 19, 50])  # 单位：M
training_time = np.array([1.5, 2.2, 1.8, 2.0])  # 单位：h
val_acc = np.array([84.2, 84.8, 83.9, 85.1])  # 单位：%
test_acc = np.array([83.1, 83.7, 82.8, 84.2])  # 单位：%
flops = np.array([4.1, 7.8, 4.5, 8.7])  # 单位：G

# 集成模型数据
ensemble_val_acc = 85.6
final_test_acc = 85.0

# 创建图形
fig = plt.figure(figsize=(16, 12))
fig.suptitle('模型性能评估与对比', fontsize=18, fontweight='bold', y=0.98)

# 子图1：准确率对比（验证集 vs 测试集）
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, val_acc, width, label='验证准确率', color='#4C72B0', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_acc, width, label='测试准确率', color='#DD8452', alpha=0.8)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

ax1.set_xlabel('模型')
ax1.set_ylabel('准确率 (%)')
ax1.set_title('验证集与测试集准确率对比', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：参数量与FLOPs对比
ax2 = plt.subplot(2, 3, 2)
x_pos = np.arange(len(models))

# 创建双y轴
ax2a = ax2
bars3 = ax2a.bar(x_pos - 0.2, parameters, 0.4, label='参数量 (M)', color='#55A868', alpha=0.7)
ax2a.set_xlabel('模型')
ax2a.set_ylabel('参数量 (M)', color='#55A868')
ax2a.tick_params(axis='y', labelcolor='#55A868')

ax2b = ax2a.twinx()
bars4 = ax2b.bar(x_pos + 0.2, flops, 0.4, label='FLOPs (G)', color='#C44E52', alpha=0.7)
ax2b.set_ylabel('FLOPs (G)', color='#C44E52')
ax2b.tick_params(axis='y', labelcolor='#C44E52')

# 添加数值标签
for bar in bars3:
    height = bar.get_height()
    ax2a.annotate(f'{height}M',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10)

for bar in bars4:
    height = bar.get_height()
    ax2b.annotate(f'{height}G',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10)

ax2.set_title('模型复杂度对比', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, rotation=45, ha='right')

# 合并图例
lines1, labels1 = ax2a.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 子图3：训练时间
ax3 = plt.subplot(2, 3, 3)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
bars5 = ax3.bar(models, training_time, color=colors, alpha=0.8)

# 添加数值标签
for bar in bars5:
    height = bar.get_height()
    ax3.annotate(f'{height}h',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

ax3.set_xlabel('模型')
ax3.set_ylabel('训练时间 (小时)')
ax3.set_title('训练时间对比', fontweight='bold')
# 修复：设置x轴刻度标签，使用正确的定位器
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, rotation=45, ha='right')
ax3.grid(True, alpha=0.3)

# 子图4：准确率 vs 参数量（散点图）
ax4 = plt.subplot(2, 3, 4)
# 修复：将s参数转换为与x,y相同长度的数组
scatter_sizes = flops * 100  # 将FLOPs转换为气泡大小
scatter = ax4.scatter(parameters, test_acc, s=scatter_sizes, c=training_time, 
                      alpha=0.7, cmap='plasma', edgecolors='black', linewidth=0.5)

# 为每个点添加模型标签
for i, model in enumerate(models):
    ax4.annotate(model, (parameters[i], test_acc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax4.set_xlabel('参数量 (M)')
ax4.set_ylabel('测试准确率 (%)')
ax4.set_title('准确率 vs 参数量（气泡大小表示FLOPs，颜色表示训练时间）', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('训练时间 (小时)')

# 子图5：集成模型效果对比
ax5 = plt.subplot(2, 3, 5)
all_models = models + ['模型集成', 'TTA+集成']
all_test_acc = list(test_acc) + [85.6, 85.0]  # 集成验证准确率和最终测试准确率
all_colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#4C72B0', '#55A868', '#C44E52']

bars6 = ax5.bar(range(len(all_models)), all_test_acc, color=all_colors, alpha=0.8)

# 添加数值标签
for bar in bars6:
    height = bar.get_height()
    ax5.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# 添加单模型最佳准确率线
best_single_acc = max(test_acc)
ax5.axhline(y=best_single_acc, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax5.text(len(all_models)-0.5, best_single_acc+0.1, f'最佳单模型: {best_single_acc}%', 
         color='red', fontsize=9, ha='right')

ax5.set_xlabel('模型/方法')
ax5.set_ylabel('准确率 (%)')
ax5.set_title('模型集成与最终效果', fontweight='bold')
ax5.set_xticks(range(len(all_models)))
ax5.set_xticklabels(all_models, rotation=45, ha='right')
ax5.grid(True, alpha=0.3)

# 子图6：性能总结雷达图
ax6 = plt.subplot(2, 3, 6, projection='polar')

# 标准化数据用于雷达图
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

categories = ['测试准确率', '训练效率', '参数量效率', '计算效率']
N = len(categories)

# 对于雷达图，我们希望值越大越好，所以对某些指标取反
test_acc_norm = normalize(test_acc)
training_efficiency = 1 / training_time  # 训练时间越短效率越高
param_efficiency = 1 / parameters  # 参数量越少效率越高
flops_efficiency = 1 / flops  # FLOPs越少计算效率越高

# 标准化所有效率指标
training_efficiency_norm = normalize(training_efficiency)
param_efficiency_norm = normalize(param_efficiency)
flops_efficiency_norm = normalize(flops_efficiency)

# 为每个模型计算综合得分
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

for i, model in enumerate(models):
    values = [test_acc_norm[i], training_efficiency_norm[i], 
              param_efficiency_norm[i], flops_efficiency_norm[i]]
    values += values[:1]  # 闭合雷达图
    
    ax6.plot(angles, values, 'o-', linewidth=2, label=model, alpha=0.7)
    ax6.fill(angles, values, alpha=0.1)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories)
ax6.set_title('模型综合性能雷达图', fontweight='bold', pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 添加数据表格
table_data = [
    ['模型', '参数量(M)', '训练时间(h)', '验证Acc(%)', '测试Acc(%)', 'FLOPs(G)'],
    ['ResNet50', 25, 1.5, 84.2, 83.1, 4.1],
    ['ResNet101', 44, 2.2, 84.8, 83.7, 7.8],
    ['EfficientNet-B4', 19, 1.8, 83.9, 82.8, 4.5],
    ['ConvNeXt-Small', 50, 2.0, 85.1, 84.2, 8.7]
]

# 在图形下方添加表格
table_ax = fig.add_axes([0.1, 0.02, 0.8, 0.1])  # 调整位置
table_ax.axis('off')
table = table_ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# 设置表格样式
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        if i == 0:  # 表头
            table[(i, j)].set_facecolor('#4C72B0')
            table[(i, j)].set_text_props(weight='bold', color='white')
        elif i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

# 保存图片
plt.savefig('模型性能评估.png', dpi=300, bbox_inches='tight')
plt.show()

# 单独创建一个集成模型权重饼图
fig2, ax = plt.subplots(figsize=(8, 8))
weights = [0.3, 0.25, 0.25, 0.2]
weight_labels = ['ConvNeXt-Small\n(30%)', 'ResNet101\n(25%)', 'ResNet50\n(25%)', 'EfficientNet-B4\n(20%)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

wedges, texts, autotexts = ax.pie(weights, labels=weight_labels, colors=colors, autopct='%1.1f%%',
                                  startangle=90, explode=(0.1, 0, 0, 0))

# 美化文本
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

ax.set_title('模型集成权重分配', fontsize=16, fontweight='bold')
plt.savefig('模型集成权重.png', dpi=300, bbox_inches='tight')
plt.show()

print("可视化已生成完成！")
print("1. 主图包含6个子图，全面展示了模型性能对比")
print("2. 单独生成了模型集成权重饼图")
print("3. 图片已保存为'模型性能评估.png'和'模型集成权重.png'")

