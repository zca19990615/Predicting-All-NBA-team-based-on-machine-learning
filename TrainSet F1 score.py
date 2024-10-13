# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:23:41 2024

@author: 61967
"""

# -*- coding: utf-8 -*- 
"""
Created on Thu Oct 10 22:51:13 2024

@author: 61967
"""

import matplotlib.pyplot as plt
import numpy as np  # Importing numpy for range adjustments

# Data setup

models = ['SVC', 'GBM', 'LR-EN', 'KNN', 'Naïve Bayes']
f1_scores = [0.8297, 0.8528, 0.7696, 0.6387, 0.6599]
colors = ['#F39B7F', '#3C5488', '#00A087', '#E64B35', '#4DBBD5']

# 定义误差范围的上下界（实际数据）
error_lower = [0.8000, 0.8200, 0.7500, 0.6000, 0.6300]
error_upper = [0.8600, 0.8800, 0.7900, 0.6700, 0.6900]

# 计算对称误差范围
errors = [(upper - lower) / 2 for lower, upper in zip(error_lower, error_upper)]

plt.figure(figsize=(10, 6))
bars = plt.barh(models, f1_scores, color=colors, xerr=errors, capsize=5, edgecolor='black', error_kw={'elinewidth': 1, 'ecolor': 'black'})

# Title and labels
plt.title('TrainSet')
plt.xlabel('F1 Score')
plt.ylabel('Models')

# Set visibility of spines to keep only the x and y axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# Set X-axis ticks
plt.xticks(np.arange(0, 1.01, 0.2))  # Fixed range from 0 to 1 with 0.2 intervals

# Enable y-axis grid lines only for clarity
plt.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

# Adding legend in the upper right corner
legend_labels = [f"{score:.4f} ({lower:.4f}-{upper:.4f})" for score, lower, upper in reversed(list(zip(f1_scores, error_lower, error_upper)))]
plt.legend(handles=reversed(bars), labels=legend_labels, title="Model F1 Score Range", loc='upper right', fontsize='small')

# Display the plot
plt.show()