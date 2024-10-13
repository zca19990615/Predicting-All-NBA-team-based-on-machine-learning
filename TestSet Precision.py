import matplotlib.pyplot as plt
import numpy as np

# 定义分类器名称和对应的精确度
classifiers_en = ['Naive Bayes', 'KNN', 'LR-EN', 'GBM', 'SVC']
precision_scores = [0.4609, 0.4112, 0.7500, 0.8333, 0.7759]
colors_new = ['#4DBBD5', '#E64B35', '#00A087', '#3C5488', '#F39B7F']  # 配色方案

# 定义误差范围的上下界（实际数据）
error_lower = [0.3675, 0.3173, 0.6334, 0.7213, 0.6596]
error_upper = [0.5508, 0.4957, 0.8636, 0.9302, 0.875]

# 计算对称误差范围
errors = [(upper - lower) / 2 for lower, upper in zip(error_lower, error_upper)]

# 创建散点图
plt.figure(figsize=(10, 6))
for i, (cls, score, err) in enumerate(zip(classifiers_en, precision_scores, errors)):
    # 绘制带误差的散点图
    plt.scatter(cls, score, color=colors_new[i], s=100, marker='o', label=f"{cls} {score:.4f} ({error_lower[i]:.4f}-{error_upper[i]:.4f})")
    plt.errorbar(cls, score, yerr=err, fmt='none', ecolor='black', elinewidth=1, capsize=5)

# 设置图表标题和标签
plt.ylabel('Precision')
plt.title('TestSet')
plt.legend(title="Model Precision Score", loc='lower right')

# 调整图表范围和网格线设置
plt.ylim(0, 1)
plt.xlim(-0.5, len(classifiers_en)-0.5)
plt.grid(True)  # 添加X轴和Y轴网格线

# 隐藏顶部和右侧的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 显示图表
plt.show()