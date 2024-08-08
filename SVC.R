# 加载必要的库
library(tidyverse) # 数据处理2023-2024.(预测版)csv
library(caret)     # 机器学习
library(e1071)     # SVM模型

data <- read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\缩减\\特征选择（26）.csv")

# 分割数据集为训练集和测试集
set.seed(123) # 确保可重复性


index <- createDataPartition(data$Injured, p = .8, list = FALSE)
trainSet <- data[index, ]
testSet <- data[-index, ]

# 转换响应变量为因子
trainSet$Injured <- as.factor(trainSet$Injured)
testSet$Injured <- as.factor(testSet$Injured)

# 使用交叉验证设置训练控制
train_control <- trainControl(method="cv", number=10)

# 训练SVC模型，包含交叉验证
svc_model_cv <- train(Injured ~ ., data = trainSet, method="svmRadial",
                      trControl=train_control, preProcess = c("center", "scale"))

# 在训练集上进行预测
train_predictions_cv <- predict(svc_model_cv, trainSet)
# 在测试集上进行预测
test_predictions_cv <- predict(svc_model_cv, testSet)

# 计算训练集的混淆矩阵及评估指标
train_conf_matrix_cv <- confusionMatrix(as.factor(train_predictions_cv), trainSet$Injured)
# 计算测试集的混淆矩阵及评估指标
test_conf_matrix_cv <- confusionMatrix(as.factor(test_predictions_cv), testSet$Injured)

# 打印训练集的混淆矩阵及评估指标
print(train_conf_matrix_cv)
print(train_conf_matrix_cv$byClass)
# 打印测试集的混淆矩阵及评估指标
print(test_conf_matrix_cv)
print(test_conf_matrix_cv$byClass)










# 加载必要的库
library(tidyverse) # 数据处理
library(caret)     # 机器学习
library(e1071)     # SVM模型

# 读取训练数据


data <- read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\缩减\\特征选择（26）.csv")

# 数据预处理步骤保持不变...

# 分割数据集为训练集和测试集
set.seed(123) # 确保可重复性


index <- createDataPartition(data$Injured, p = .8, list = FALSE)
trainSet <- data[index, ]
testSet <- data[-index, ]
# 转换响应变量为因子
trainSet$Injured <- as.factor(trainSet$Injured)

# 使用交叉验证设置训练控制
train_control <- trainControl(method="cv", number=10)

# 训练SVC模型，包含交叉验证
svc_model <- train(Injured ~ ., data = trainSet, method="svmRadial",
                   trControl=train_control, preProcess = c("center", "scale"))

# 读取测试数据
# 注意：请确保测试数据文件路径和文件名是正确的
test_data <- read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\2023-2024.(1)csv.csv") # 此处替换为实际测试数据文件的路径和文件名

# 假设测试数据集也包含'Injured'列，即实际结果，需要在预测之前将其转为因子类型
test_data$Injured <- as.factor(test_data$Injured)

# 保存测试集的实际结果，以便与预测结果进行比较
actual_outcomes <- test_data$Injured

# 移除测试数据集的'Injured'列，因为这是我们要预测的
test_data <- test_data %>% select(-Injured)

# 使用训练好的模型在测试数据上进行预测
test_predictions <- predict(svc_model, test_data)

# 创建一个数据框以比较预测结果和实际结果
# 假设测试数据集中有一个名为'Player'的列，它包含球员的名字
prediction_results <- data.frame(
  Player = test_data$PER, # 请替换为测试数据集中包含球员名字的字段名
  Predicted = as.numeric(test_predictions),
  Outcome = as.numeric(actual_outcomes),
  All_NBA_Team = ifelse(actual_outcomes == "1", "Yes", "No")
)

# 计算混淆矩阵及评估指标
conf_matrix <- confusionMatrix(as.factor(test_predictions), actual_outcomes)

# 打印混淆矩阵及评估指标
print(conf_matrix)
print(conf_matrix$byClass)

# 查看预测结果数据框的头部
head(prediction_results)


# 打印完整的预测结果数据框
print(prediction_results)

