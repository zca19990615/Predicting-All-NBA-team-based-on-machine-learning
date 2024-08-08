import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import numpy as np

# 加载数据
data = pd.read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\total2.csv")

# 移除非数值列
numeric_data = data.select_dtypes(include=np.number)

# 目标变量
target = 'Injured'

# 设置随机森林分类器
model = RandomForestClassifier()

# 设置交叉验证方法：重复的10折交叉验证
cv = RepeatedKFold(n_splits=10, n_repeats=3)

# 递归特征消除与交叉验证（RFE-CV）
selector = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy')

# 适应模型
X = numeric_data.drop(columns=[target])
y = numeric_data[target]
selector = selector.fit(X, y)

# 查看结果
print(f"Optimal number of features : {selector.n_features_}")
print('Best features :', X.columns[selector.support_])

# 选出最好的特征集
optimal_features = X.columns[selector.support_].tolist()
print(optimal_features)










