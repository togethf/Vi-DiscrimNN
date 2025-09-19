import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 示例数据
categories = ['LAPD', 'voc12', 'coco']

data = {
    'DACE_n': [0.148, 0.192, 0.201],
    'DACE_m': [0.038, 0.149, 0.153],   
    'DACE_l': [0.049, 0.155, 0.147],
    'mAPI_n': [0.111, 0.106, 0.159],
    'mAPI_m': [-0.041, 0.023, 0.058],
    'mAPI_l': [-0.027, 0, 0.04]
}
scale = 100

# 自定义配色方案（基于主题的渐变色彩）
positive_colors = ['#3498db', '#2980b9', '#1a5276']
negative_colors = ['#e74c3c', '#c0392b', '#943126']

group_colors = {
    'DACE_n': positive_colors[0],
    'DACE_m': positive_colors[1],
    'DACE_l': positive_colors[2],
    'mAPI_n': negative_colors[0],
    'mAPI_m': negative_colors[1],
    'mAPI_l': negative_colors[2],
}

# 设置图形参数
categories = np.array(categories)
x = np.arange(len(categories))

# 合理设置柱宽和偏移
num_bars_per_group = len(data) + 7
total_width = 2.15 # 每组柱子在一个x位置所占的总宽度
width = total_width / num_bars_per_group

# 创建图形和坐标轴
fig, ax = plt.subplots(layout='constrained', figsize=(16, 9), dpi=300)

# 背景设置
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 绘制分组柱状图
for i, (attribute, measurement) in enumerate(data.items()):
    offset = (i-2 - num_bars_per_group / 2) * width + width / 2
    measurement = [m * scale for m in measurement]
    bars = ax.bar(x + offset, measurement, width, label=attribute, color=group_colors[attribute],
                  edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3)
    
    # 为每个柱子添加数值标签
    for rect, value in zip(bars, measurement):
        height = rect.get_height()
        ax.annotate(f'{value:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4 if height >= 0 else -4),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=24)

# 设置标签和标题
ax.set_ylabel('Δ Performance', fontsize=24, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=24)

legend = ax.legend(loc='upper right', 
                   frameon=True, edgecolor='black', framealpha=0.9,
                   title='Models', fontsize=26,markerscale=0.7)
legend.get_title().set_fontweight('bold')

# 添加零线参考线
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray', zorder=0)

# 设置y轴范围
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min - 0.02, y_max + 0.02)

# 添加边框
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1)

# 设置Y轴刻度字体大小
plt.yticks(fontsize=24)

# 保存图像
plt.savefig('./experiments/difficulty_diff.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()