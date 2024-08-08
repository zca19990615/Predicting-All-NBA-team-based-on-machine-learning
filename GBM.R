library(randomForest)
library(tidyverse) # 数据处理
library(caret)     # 机器学习
library(e1071)     # SVM模型，可用于比较
library(tidyverse) # 数据处理
library(caret)     # 机器学习
library(gbm)       # GBM模型
library(doParallel)
library(pROC)
registerDoParallel(cores = detectCores())

library(doParallel)
registerDoParallel(cores=detectCores())
# 假设您的数据框为data，响应变量列为Injured
set.seed(123)  # 设置随机种子以保证可重复性

# 读取数据

data <- read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\缩减\\特征选择（26）.csv")
# 清理列名（移除特殊字符等）
clean_colnames <- colnames(data) %>%
  gsub("%", "Percent", .) %>%
  gsub("^3P$", "ThreeP", .) %>%
  gsub("^3PA$", "ThreePA", .) %>%
  gsub("[^[:alnum:]_]", "", .) %>%
  gsub("^[0-9]", "X", .)
colnames(data) <- clean_colnames

# 分割数据集为训练集和测试集
set.seed(123) # 确保可重复性
index <- createDataPartition(data$Injured, p = .8, list = FALSE)
trainSet <- data[index, ]
testSet <- data[-index, ]

# 使用GBM训练模型
gbm_model <- gbm(Injured ~ ., data = trainSet, distribution = "bernoulli", n.trees = 500, interaction.depth = 4, shrinkage = 0.01, cv.folds = 10)

# 查看模型的摘要信息
summary(gbm_model)

# 使用模型进行预测
prediction_probabilities <- predict(gbm_model, testSet, type = "response", n.trees = gbm_model$n.trees)

# 计算ROC曲线和AUC
roc_result <- roc(testSet$Injured, prediction_probabilities)
auc_value <- auc(roc_result)
print(auc_value)

# 新代码：在训练集上进行预测，为训练数据创建混淆矩阵
train_predictions <- predict(gbm_model, trainSet, type = "response", n.trees = gbm_model$n.trees)
train_predictions_binary <- ifelse(train_predictions > 0.5, 1, 0) # 将概率转换为二进制结果
train_conf_matrix <- confusionMatrix(as.factor(train_predictions_binary), as.factor(trainSet$Injured))


# 显示训练集的混淆矩阵
print(train_conf_matrix)
print(train_conf_matrix$byClass)





# 计算准确度、精确度、召回率和F1分数
predictions <- ifelse(prediction_probabilities > 0.5, 1, 0) # 选择一个阈值，例如0.5
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(testSet$Injured))




# 查看混淆矩阵和相关指标
print(conf_matrix)
print(conf_matrix$byClass)





# 计算MSE和R²
mse <- mean((predictions - testSet$Injured)^2)
r_squared <- 1 - (sum((predictions - testSet$Injured)^2) / sum((mean(trainSet$Injured) - testSet$Injured)^2))
print(mse)
print(r_squared)

