# 加载必要的库
library(tidyverse) # 数据处理
library(caret)     # 机器学习
library(pROC)      # 用于计算AUC和置信区间
library(boot)      # 引入bootstrap用于置信区间计算
library(ggplot2)   # 用于绘图
library(glmnet)    # 用于Elastic Net模型

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

# 训练Elastic Net模型，并使用交叉验证
set.seed(123)
elastic_net_model <- train(Injured ~ ., data = trainSet, method="glmnet", trControl=train_control, tuneLength=5, family="binomial")

# 在训练集上进行预测
train_predictions <- predict(elastic_net_model, trainSet)

# 显式定义正类，避免混淆
train_conf_matrix <- confusionMatrix(train_predictions, trainSet$Injured, positive = "1")  # 确保定义正确
print("Training Set Confusion Matrix and Metrics:")
print(train_conf_matrix)
print("Train Set Classification Report:")
print(train_conf_matrix$byClass)

# 在测试集上进行预测
test_predictions <- predict(elastic_net_model, testSet)

# 显式定义正类，避免混淆
test_conf_matrix <- confusionMatrix(test_predictions, testSet$Injured, positive = "1")  # 确保定义正确
print("Testing Set Confusion Matrix and Metrics:")
print(test_conf_matrix)
print("Test Set Classification Report:")
print(test_conf_matrix$byClass)

# 手动计算主要指标，确保与自动计算一致
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

# 计算AUC
roc_obj <- roc(response = testSet$Injured, predictor = as.numeric(as.character(test_predictions)))

# 计算AUC的置信区间
auc_ci <- ci(roc_obj)

# 打印AUC的置信区间
print(paste("AUC 95% CI:", auc_ci[1], "-", auc_ci[3]))

# 使用 Bootstrap 来计算测试集指标的置信区间
bootstrap_metrics <- function(data, indices) {
  boot_sample <- data[indices, ]
  
  # 进行预测
  predictions <- predict(elastic_net_model, boot_sample)
  
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

# 计算95%置信区间
accuracy_ci <- boot.ci(boot_results, type="perc", index=1)
precision_ci <- boot.ci(boot_results, type="perc", index=2)
sensitivity_ci <- boot.ci(boot_results, type="perc", index=3)
specificity_ci <- boot.ci(boot_results, type="perc", index=4)
f1_ci <- boot.ci(boot_results, type="perc", index=5)

# 打印95%置信区间
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
  labs(title = "Elastic Net Model Performance with 95% Confidence Intervals",
       y = "Metric Value", x = "Performance Metric") +
  theme_minimal()

