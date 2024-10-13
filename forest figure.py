import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
# 模型名称、平均AUC和置信区间数据
##model_names = ["Naive Bayers","KNN", "LR-EN", "GBM", "SVC"]
##mean_aucs = [0.9045, 0.9646, 0.9565, 0.971, 0.9684]
##conf_intervals = [(0.8919, 0.9161), (0.9563, 0.9717), (0.9474, 0.9644), (0.9634, 0.9774), (0.9605, 0.9751)]
##errors = [(mean - conf[0], conf[1] - mean) for mean, conf in zip(mean_aucs, conf_intervals)]
##colors = ['#FF6347', '#4169E1', '#FFD700', '#32CD32', '#9370DB']


model_names = ["Naive Bayers","KNN", "LR-EN", "GBM", "SVC"]
mean_aucs = [0.9002, 0.9334, 0.9574, 0.9710, 0.9667]
conf_intervals = [(0.8874, 0.9121), (0.9101, 0.9552), (0.9484, 0.9652), (0.9634, 0.9774), (0.9587, 0.9736)]
errors = [(mean - conf[0], conf[1] - mean) for mean, conf in zip(mean_aucs, conf_intervals)]
colors = ['#FF6347', '#4169E1', '#FFD700', '#32CD32', '#9370DB']

# 开始绘制森林图
fig, ax = plt.subplots(figsize=(10, 6))



for i, (mean, error) in enumerate(zip(mean_aucs, errors)):
    ax.errorbar(i, mean, yerr=np.array([error]).T, fmt='o', color=colors[i], 
                markersize=10, elinewidth=2, capsize=5, label=model_names[i])
# 设置X轴为模型名称，Y轴为AUC分数
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_yticks(np.arange(0.80, 1.05, 0.05))
ax.set_ylabel('Accuracy Score')
ax.set_title('TrainSet')

# 移除网格线
ax.grid(False)

# 准备绘制信息框内容
# 准备绘制信息框内容
legend_patches = [
    mpatches.Patch(
        color=color,
        label=f'{model}: {mean_auc:.4f} ({conf[0]:.4f}-{conf[1]:.4f})'
    )
    for color, model, mean_auc, conf in zip(colors, model_names, mean_aucs, conf_intervals)
]

# 绘制信息框
legend = ax.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1, 0), 
                   frameon=True, framealpha=0.5, title='Model Accuracy Score')
legend.get_title().set_fontsize('10')

# 隐藏Y轴的刻度线
ax.yaxis.set_ticks_position('none') 

# 调整布局以防止X轴标签被剪切
plt.tight_layout()

# 显示图形
plt.show()


