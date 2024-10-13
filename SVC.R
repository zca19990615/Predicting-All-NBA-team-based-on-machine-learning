# 加载必要的库
library(tidyverse) # 数据处理
library(caret)     # 机器学习
library(gbm)       # GBM模型
library(boot)      # Bootstrap用于计算置信区间
library(ggplot2)   # 用于绘图
library(pROC)      # 用于计算AUC和置信区间
library(glmnet)    # 用于Elastic Net模型
library(e1071)     # 用于朴素贝叶斯模型

# 读取训练数据
data <- read_csv("C:\Users\61967\Desktop\机器学习2\new\缩减\特征选择（26）.csv")

# 分割数据集为训练集和测试集
set.seed(123) # 确保可重复性
index <- createDataPartition(data$Injured, p = .8, list = FALSE)
trainSet <- data[index, ]
testSet <- data[-index, ]

# 转换响应变量为因子
trainSet$Injured <- as.factor(trainSet$Injured)
testSet$Injured <- as.factor(testSet$Injured)

# 使用交叉验证设置训练控制
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)

# 训练朴素贝叶斯模型
nb_model <- train(Injured ~ ., data = trainSet, method="naive_bayes", trControl=train_control)

# 在训练数据集上进行预测
train_predictions_nb <- predict(nb_model, trainSet)

# 计算混淆矩阵及评估指标（朴素贝叶斯模型 - 训练集）
train_conf_matrix_nb <- confusionMatrix(train_predictions_nb, trainSet$Injured)
print("Naive Bayes Model - Training Set Confusion Matrix and Metrics:")
print(train_conf_matrix_nb)
print("Naive Bayes Model - Train Set Classification Report:")
print(train_conf_matrix_nb$byClass)

# Bootstrap 来计算训练集指标的置信区间（朴素贝叶斯模型）
bootstrap_metrics_train_nb <- function(data, indices) {
  # 创建bootstrap样本
  boot_sample <- data[indices, ]
  
  # 在bootstrap样本上进行预测
  predictions <- predict(nb_model, boot_sample)
  
  # 计算混淆矩阵
  conf_matrix <- confusionMatrix(predictions, boot_sample$Injured)
  
  # 提取各项指标
  accuracy <- conf_matrix$overall["Accuracy"]
  precision <- conf_matrix$byClass["Pos Pred Value"]
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  f1 <- (2 * precision * sensitivity) / (precision + sensitivity)
  
  return(c(Accuracy = accuracy, Precision = precision, Sensitivity = sensitivity, Specificity = specificity, F1 = f1))
}

# 使用 boot 函数对训练集进行1000次bootstrap（朴素贝叶斯模型）
set.seed(123)
train_nb_boot_results <- boot(data = trainSet, statistic = bootstrap_metrics_train_nb, R = 1000)

# 计算95%置信区间（朴素贝叶斯模型 - 训练集）
train_nb_accuracy_ci <- boot.ci(train_nb_boot_results, type="perc", index=1)
train_nb_precision_ci <- boot.ci(train_nb_boot_results, type="perc", index=2)
train_nb_sensitivity_ci <- boot.ci(train_nb_boot_results, type="perc", index=3)
train_nb_specificity_ci <- boot.ci(train_nb_boot_results, type="perc", index=4)
train_nb_f1_ci <- boot.ci(train_nb_boot_results, type="perc", index=5)

# 打印训练集的置信区间（朴素贝叶斯模型）
print(paste("Naive Bayes Model - Training Set 95% CI for Accuracy:", train_nb_accuracy_ci$percent[4], "-", train_nb_accuracy_ci$percent[5]))
print(paste("Naive Bayes Model - Training Set 95% CI for Precision:", train_nb_precision_ci$percent[4], "-", train_nb_precision_ci$percent[5]))
print(paste("Naive Bayes Model - Training Set 95% CI for Sensitivity:", train_nb_sensitivity_ci$percent[4], "-", train_nb_sensitivity_ci$percent[5]))
print(paste("Naive Bayes Model - Training Set 95% CI for Specificity:", train_nb_specificity_ci$percent[4], "-", train_nb_specificity_ci$percent[5]))
print(paste("Naive Bayes Model - Training Set 95% CI for F1 Score:", train_nb_f1_ci$percent[4], "-", train_nb_f1_ci$percent[5]))



