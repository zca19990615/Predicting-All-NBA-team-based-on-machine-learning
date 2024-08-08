# 加载必要的库
library(tidyverse) # 数据处理
library(caret)     # 机器学习
library(e1071)     # 包含朴素贝叶斯模型


# 读取数据
data <- read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\缩减\\特征选择（26）.csv")

# 数据预处理步骤保持不变...

# 分割数据集为训练集和测试集
set.seed(123) # 确保可重复性

# 使用原始数据框进行分割
index <- createDataPartition(data$Injured, p = .8, list = FALSE)
trainSet <- data[index, ]
testSet <- data[-index, ]

# 转换响应变量为因子
trainSet$Injured <- as.factor(trainSet$Injured)
testSet$Injured <- as.factor(testSet$Injured)

# 设置交叉验证
train_control <- trainControl(method="cv", number=10) # 10-fold cross-validation

# 训练朴素贝叶斯模型，并使用交叉验证
nb_model <- train(Injured ~ ., data = trainSet, method="naive_bayes", trControl=train_control)

# 在训练集上进行预测
train_predictions <- predict(nb_model, trainSet)
# 在测试集上进行预测
test_predictions <- predict(nb_model, testSet)

# 计算训练集的混淆矩阵及评估指标
train_conf_matrix <- confusionMatrix(as.factor(train_predictions), trainSet$Injured)
# 计算测试集的混淆矩阵及评估指标
test_conf_matrix <- confusionMatrix(as.factor(test_predictions), testSet$Injured)

# 打印训练集的混淆矩阵及评估指标
print(train_conf_matrix)
print(train_conf_matrix$byClass)
# 打印测试集的混淆矩阵及评估指标
print(test_conf_matrix)
print(test_conf_matrix$byClass)
# 打印交叉验证的结果
print(nb_model)

