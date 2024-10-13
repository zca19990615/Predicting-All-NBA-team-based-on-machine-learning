# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:57:33 2024

@author: 61967
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 定义混淆矩阵数据
confusion_matrices = {
    "KNN": np.array([[521, 41], [7, 17]]),
    "SVC": np.array([[519, 13], [9, 45]]),
    "LR-EN": np.array([[511, 17], [15, 43]]),
    "Naive Bayes": np.array([[467, 5], [61, 53]]),
    "GBM": np.array([[514, 13], [14, 45]])
}

# 函数用于绘制并保存混淆矩阵图
def plot_and_save_confusion_matrix(matrix, title, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Reds", cbar=True)
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(output_path)
    plt.close()

# 生成和保存每个模型的混淆矩阵图
output_paths = {
    "KNN": "C:/Users/61967/Desktop/机器学习2/算法/终稿/二次返修/代码/图/矩阵/knn_confusion_matrix_new.png",
    "SVC": "C:/Users/61967/Desktop/机器学习2/算法/终稿/二次返修/代码/图/矩阵/svc_confusion_matrix_new.png",
    "LR-EN": "C:/Users/61967/Desktop/机器学习2/算法/终稿/二次返修/代码/图/矩阵/lr_en_confusion_matrix_new.png",
    "Naive Bayes": "C:/Users/61967/Desktop/机器学习2/算法/终稿/二次返修/代码/图/矩阵/nb_confusion_matrix_new.png",
    "GBM": "C:/Users/61967/Desktop/机器学习2/算法/终稿/二次返修/代码/图/矩阵/gbm_confusion_matrix_new.png"
}

for model, matrix in confusion_matrices.items():
    plot_and_save_confusion_matrix(matrix, model, output_paths[model])

# 输出路径
output_paths
