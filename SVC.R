# 加载必要的库
library(tidyverse) # 数据处理
library(caret)     # 机器学习
library(pROC)      # 用于计算AUC和置信区间
library(boot)      # 引入bootstrap用于置信区间计算
library(ggplot2)   # 用于绘图
library(e1071)     # 用于SVM模型

# 读取数据
data <- read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\缩减\\特征选择（26）.csv")

# 分割数据集为训练集和测试集
set.seed(123) # 确保可重复性
index <- createDataPartition(data$Injured, p = .8, list = FALSE)
trainSet <- data[index, ]
testSet <- data[-index, ]

# 转换响应变量为因子
trainSet$Injured <- as.factor(trainSet$Injured)
testSet$Injured <- as.factor(testSet$Injured)

# 设置交叉验证
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)  # 10-fold cross-validation

# 训练SVM模型，并使用交叉验证
set.seed(123)
svc_model_cv <- train(Injured ~ ., data = trainSet, method="svmRadial", trControl=train_control, preProcess=c("center", "scale"))

# 在训练集上进行预测
train_predictions <- predict(svc_model_cv, trainSet)

# 显式定义正类，避免混淆
train_conf_matrix <- confusionMatrix(train_predictions, trainSet$Injured, positive = "1")  # 确保定义正确
print("Training Set Confusion Matrix and Metrics:")
print(train_conf_matrix)
print("Train Set Classification Report:")
print(train_conf_matrix$byClass)

# 手动计算训练集主要指标
train_tp <- sum(train_predictions == "1" & trainSet$Injured == "1")
train_tn <- sum(train_predictions == "0" & trainSet$Injured == "0")
train_fp <- sum(train_predictions == "1" & trainSet$Injured == "0")
train_fn <- sum(train_predictions == "0" & trainSet$Injured == "1")

train_sensitivity <- train_tp / (train_tp + train_fn)
train_specificity <- train_tn / (train_tn + train_fp)
train_precision <- train_tp / (train_tp + train_fp)
train_f1 <- (2 * train_precision * train_sensitivity) / (train_precision + train_sensitivity)

print(paste("Manual Training Sensitivity:", train_sensitivity))
print(paste("Manual Training Specificity:", train_specificity))
print(paste("Manual Training Precision:", train_precision))
print(paste("Manual Training F1 Score:", train_f1))

# 使用 Bootstrap 来计算训练集指标的置信区间
bootstrap_train_metrics <- function(data, indices) {
  boot_sample <- data[indices, ]
  
  # 进行预测
  predictions <- predict(svc_model_cv, boot_sample)
  
  # 计算混淆矩阵
  conf_matrix <- confusionMatrix(predictions, boot_sample$Injured, positive = "1")  # 确保定义一致
  
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
boot_train_results <- boot(data = trainSet, statistic = bootstrap_train_metrics, R = 1000)

# 计算训练集95%置信区间
train_accuracy_ci <- boot.ci(boot_train_results, type="perc", index=1)
train_precision_ci <- boot.ci(boot_train_results, type="perc", index=2)
train_sensitivity_ci <- boot.ci(boot_train_results, type="perc", index=3)
train_specificity_ci <- boot.ci(boot_train_results, type="perc", index=4)
train_f1_ci <- boot.ci(boot_train_results, type="perc", index=5)

# 打印训练集95%置信区间
print(paste("Training Set 95% CI for Accuracy:", train_accuracy_ci$percent[4], "-", train_accuracy_ci$percent[5]))
print(paste("Training Set 95% CI for Precision:", train_precision_ci$percent[4], "-", train_precision_ci$percent[5]))
print(paste("Training Set 95% CI for Sensitivity:", train_sensitivity_ci$percent[4], "-", train_sensitivity_ci$percent[5]))
print(paste("Training Set 95% CI for Specificity:", train_specificity_ci$percent[4], "-", train_specificity_ci$percent[5]))
print(paste("Training Set 95% CI for F1 Score:", train_f1_ci$percent[4], "-", train_f1_ci$percent[5]))

# 在测试集上进行预测
test_predictions <- predict(svc_model_cv, testSet)

# 显式定义正类，避免混淆
test_conf_matrix <- confusionMatrix(test_predictions, testSet$Injured, positive = "1")  # 确保定义正确
print("Testing Set Confusion Matrix and Metrics:")
print(test_conf_matrix)
print("Test Set Classification Report:")
print(test_conf_matrix$byClass)

# 手动计算测试集主要指标
tp <- sum(test_predictions == "1" & testSet$Injured == "1")
tn <- sum(test_predictions == "0" & testSet$Injured == "0")
fp <- sum(test_predictions == "1" & testSet$Injured == "0")
fn <- sum(test_predictions == "0" & testSet$Injured == "1")

sensitivity <- tp / (tp + fn)
specificity <- tn / (tn + fp)
precision <- tp / (tp + fp)
f1 <- (2 * precision * sensitivity) / (precision + sensitivity)

print(paste("Manual Sensitivity:", sensitivity))
print(paste("Manual Specificity:", specificity))
print(paste("Manual Precision:", precision))
print(paste("Manual F1 Score:", f1))

# 使用 Bootstrap 来计算测试集指标的置信区间
bootstrap_metrics <- function(data, indices) {
  boot_sample <- data[indices, ]
  
  # 进行预测
  predictions <- predict(svc_model_cv, boot_sample)
  
  # 计算混淆矩阵
  conf_matrix <- confusionMatrix(predictions, boot_sample$Injured, positive = "1")  # 确保定义一致
  
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

# 计算测试集95%置信区间
accuracy_ci <- boot.ci(boot_results, type="perc", index=1)
precision_ci <- boot.ci(boot_results, type="perc", index=2)
sensitivity_ci <- boot.ci(boot_results, type="perc", index=3)
specificity_ci <- boot.ci(boot_results, type="perc", index=4)
f1_ci <- boot.ci(boot_results, type="perc", index=5)

# 打印测试集95%置信区间
print(paste("95% CI for Accuracy:", accuracy_ci$percent[4], "-", accuracy_ci$percent[5]))
print(paste("95% CI for Precision:", precision_ci$percent[4], "-", precision_ci$percent[5]))
print(paste("95% CI for Sensitivity:", sensitivity_ci$percent[4], "-", sensitivity_ci$percent[5]))
print(paste("95% CI for Specificity:", specificity_ci$percent[4], "-", specificity_ci$percent[5]))
print(paste("95% CI for F1 Score:", f1_ci$percent[4], "-", f1_ci$percent[5]))

# 准备绘图数据
metrics <- c("Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score")
train_means <- c(mean(boot_train_results$t[, 1]), mean(boot_train_results$t[, 2]), mean(boot_train_results$t[, 3]), mean(boot_train_results$t[, 4]), mean(boot_train_results$t[, 5]))
test_means <- c(mean(boot_results$t[, 1]), mean(boot_results$t[, 2]), mean(boot_results$t[, 3]), mean(boot_results$t[, 4]), mean(boot_results$t[, 5]))
train_ci_lower <- c(train_accuracy_ci$percent[4], train_precision_ci$percent[4], train_sensitivity_ci$percent[4], train_specificity_ci$percent[4], train_f1_ci$percent[4])
train_ci_upper <- c(train_accuracy_ci$percent[5], train_precision_ci$percent[5], train_sensitivity_ci$percent[5], train_specificity_ci$percent[5], train_f1_ci$percent[5])
test_ci_lower <- c(accuracy_ci$percent[4], precision_ci$percent[4], sensitivity_ci$percent[4], specificity_ci$percent[4], f1_ci$percent[4])
test_ci_upper <- c(accuracy_ci$percent[5], precision_ci$percent[5], sensitivity_ci$percent[5], specificity_ci$percent[5], f1_ci$percent[5])
plot_data <- data.frame(Metric = rep(metrics, 2), Set = rep(c("Train", "Test"), each = 5), Mean = c(train_means, test_means), CI_Lower = c(train_ci_lower, test_ci_lower), CI_Upper = c(train_ci_upper, test_ci_upper))

# 使用ggplot2绘制置信区间图
ggplot(plot_data, aes(x = Metric, y = Mean, color = Set)) +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.2, position = position_dodge(width = 0.5)) +
  labs(title = "SVM Model Performance with 95% Confidence Intervals",
       y = "Metric Value", x = "Performance Metric") +
  theme_minimal()


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




