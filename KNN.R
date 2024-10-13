# 加载必要的库
library(tidyverse) # 数据处理
library(caret)     # 机器学习
library(gbm)       # GBM模型
library(boot)      # Bootstrap用于计算置信区间
library(ggplot2)   # 用于绘图

# 读取训练数据
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
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)

# 训练GBM模型，包含交叉验证
gbm_model <- train(Injured ~ ., data = trainSet, method="gbm",
                   trControl=train_control, preProcess = c("center", "scale"),
                   verbose = FALSE)

# 在训练数据集上进行预测
train_predictions <- predict(gbm_model, trainSet)

# 计算训练集混淆矩阵及评估指标
train_conf_matrix <- confusionMatrix(as.factor(train_predictions), trainSet$Injured)
print("Training Set Confusion Matrix and Metrics:")
print(train_conf_matrix)
print("Training Set Classification Report:")
print(train_conf_matrix$byClass)

# Bootstrap 来计算训练集指标的置信区间
bootstrap_metrics_train <- function(data, indices) {
  # 使用提供的 indices 从训练集中创建一个 bootstrap 样本
  boot_sample <- data[indices, ]
  
  # 在 bootstrap 样本上进行预测
  predictions <- predict(gbm_model, boot_sample)
  
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

# 使用 boot 函数对训练集进行1000次bootstrap
set.seed(123)
boot_results_train <- boot(data = trainSet, statistic = bootstrap_metrics_train, R = 1000)

# 计算95%置信区间（训练集）
train_accuracy_ci <- boot.ci(boot_results_train, type="perc", index=1)
train_precision_ci <- boot.ci(boot_results_train, type="perc", index=2)
train_sensitivity_ci <- boot.ci(boot_results_train, type="perc", index=3)
train_specificity_ci <- boot.ci(boot_results_train, type="perc", index=4)
train_f1_ci <- boot.ci(boot_results_train, type="perc", index=5)

# 打印训练集置信区间
print(paste("95% CI for Training Accuracy:", train_accuracy_ci$percent[4], "-", train_accuracy_ci$percent[5]))
print(paste("95% CI for Training Precision:", train_precision_ci$percent[4], "-", train_precision_ci$percent[5]))
print(paste("95% CI for Training Sensitivity:", train_sensitivity_ci$percent[4], "-", train_sensitivity_ci$percent[5]))
print(paste("95% CI for Training Specificity:", train_specificity_ci$percent[4], "-", train_specificity_ci$percent[5]))
print(paste("95% CI for Training F1 Score:", train_f1_ci$percent[4], "-", train_f1_ci$percent[5]))

# 在测试数据集上进行预测
test_predictions <- predict(gbm_model, testSet)

# 计算测试集混淆矩阵及评估指标
test_conf_matrix <- confusionMatrix(as.factor(test_predictions), testSet$Injured)
print("Testing Set Confusion Matrix and Metrics:")
print(test_conf_matrix)
print("Test Set Classification Report:")
print(test_conf_matrix$byClass)

# 手动计算主要指标，确保与自动计算一致
test_sensitivity <- test_conf_matrix$byClass["Sensitivity"]
test_specificity <- test_conf_matrix$byClass["Specificity"]
test_precision <- test_conf_matrix$byClass["Pos Pred Value"]
test_recall <- test_conf_matrix$byClass["Sensitivity"]
test_f1 <- (2 * test_precision * test_recall) / (test_precision + test_recall)

print(paste("Manual Sensitivity:", test_sensitivity))
print(paste("Manual Specificity:", test_specificity))
print(paste("Manual Precision:", test_precision))
print(paste("Manual F1 Score:", test_f1))

# Bootstrap 来计算测试集指标的置信区间
bootstrap_metrics <- function(data, indices) {
  # 使用提供的 indices 从测试集中创建一个 bootstrap 样本
  boot_sample <- data[indices, ]
  
  # 在 bootstrap 样本上进行预测
  predictions <- predict(gbm_model, boot_sample)
  
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

# 使用 boot 函数对测试集进行1000次bootstrap
set.seed(123)
boot_results <- boot(data = testSet, statistic = bootstrap_metrics, R = 1000)

# 计算95%置信区间
accuracy_ci <- boot.ci(boot_results, type="perc", index=1)
precision_ci <- boot.ci(boot_results, type="perc", index=2)
sensitivity_ci <- boot.ci(boot_results, type="perc", index=3)
specificity_ci <- boot.ci(boot_results, type="perc", index=4)
f1_ci <- boot.ci(boot_results, type="perc", index=5)

# 打印置信区间
print(paste("95% CI for Accuracy:", accuracy_ci$percent[4], "-", accuracy_ci$percent[5]))
print(paste("95% CI for Precision:", precision_ci$percent[4], "-", precision_ci$percent[5]))
print(paste("95% CI for Sensitivity:", sensitivity_ci$percent[4], "-", sensitivity_ci$percent[5]))
print(paste("95% CI for Specificity:", specificity_ci$percent[4], "-", specificity_ci$percent[5]))
print(paste("95% CI for F1 Score:", f1_ci$percent[4], "-", f1_ci$percent[5]))

# 准备绘图数据
metrics <- c("Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score")
means <- c(mean(boot_results$t[, 1]), mean(boot_results$t[, 2]), mean(boot_results$t[, 3]), mean(boot_results$t[, 4]), mean(boot_results$t[, 5]))
ci_lower <- c(accuracy_ci$percent[4], precision_ci$percent[4], sensitivity_ci$percent[4], specificity_ci$percent[4], f1_ci$percent[4])
ci_upper <- c(accuracy_ci$percent[5], precision_ci$percent[5], sensitivity_ci$percent[5], specificity_ci$percent[5], f1_ci$percent[5])

plot_data <- data.frame(Metric = metrics, Mean = means, CI_Lower = ci_lower, CI_Upper = ci_upper)

# 使用ggplot2绘制置信区间图
ggplot(plot_data, aes(x = Metric, y = Mean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.2) +
  labs(title = "GBM Model Performance with 95% Confidence Intervals",
       y = "Metric Value", x = "Performance Metric") +
  theme_minimal()

# 读取新测试数据
test_data <- read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\2023-2024.(1)csv.csv") # 此处替换为实际测试数据文件的路径和文件名

# 假设测试数据集也包含'Injured'列，即实际结果，需要在预测之前将其转为因子类型
test_data$Injured <- as.factor(test_data$Injured)

# 保存测试集的实际结果，以便与预测结果进行比较
actual_outcomes <- test_data$Injured

# 移除测试数据集的'Injured'列，因为这是我们要预测的
test_data <- test_data %>% select(-Injured)

# 使用训练好的模型在测试数据上进行预测
test_predictions <- predict(gbm_model, test_data)

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
print("New Test Set Confusion Matrix and Metrics:")
print(conf_matrix)
print(conf_matrix$byClass)

# 查看预测结果数据框的头部
print("Prediction Results Head:")
print(head(prediction_results))

# 打印完整的预测结果数据框
print("Complete Prediction Results:")
print(prediction_results)
