# 加载所需的库
library(readr) # 用于读取CSV文件
library(glmnet) # 用于Elastic Net
library(caret) # 机器学习和数据划分

# 读取数据


data <- read_csv("C:\\Users\\61967\\Desktop\\机器学习2\\new\\缩减\\特征选择（26）.csv")

# 数据预处理
# 更改列名以避免特殊字符和百分比符号
colnames(data) <- gsub("%", "Percent", colnames(data))
colnames(data) <- gsub("^3P$", "ThreeP", colnames(data))
colnames(data) <- gsub("^3PA$", "ThreePA", colnames(data))
# 移除所有特殊字符并处理数字开头的列名
clean_colnames <- gsub("[^[:alnum:]_]", "", colnames(data))
clean_colnames <- ifelse(grepl("^[0-9]", clean_colnames), paste0("X", clean_colnames), clean_colnames)
colnames(data) <- clean_colnames

# 将响应变量转换为因子类型，并且更改类别标签以符合R变量命名规则
data$Injured <- factor(data$Injured, levels = c("0", "1"), labels = c("Class0", "Class1"))

# 使用createDataPartition分割数据集
index <- createDataPartition(data$Injured, p = .8, list = FALSE)
trainSet <- data[index, ]
testSet <- data[-index, ]

# 准备训练数据和测试数据
x_train <- as.matrix(trainSet[, -which(names(trainSet) == "Injured")])
y_train <- trainSet$Injured
x_test <- as.matrix(testSet[, -which(names(testSet) == "Injured")])
y_test <- testSet$Injured

# 定义训练控制，用于指定交叉验证的方式
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# 使用caret的train函数进行Elastic Net训练，包括交叉验证
set.seed(123) # 设置随机种子
elastic_net_model <- train(x_train, y_train, method = "glmnet",
                           trControl = train_control,
                           tuneLength = 5,
                           metric = "ROC",
                           family = "binomial")

# 输出模型详细信息
print(elastic_net_model)


# 使用最佳模型进行训练集预测，并获取正类的概率
train_predictions <- predict(elastic_net_model, newdata = x_train, type = "prob")[, "Class1"]
train_predicted_classes <- ifelse(train_predictions > 0.5, "Class1", "Class0")

# 生成和评估训练集上的混淆矩阵
train_confusionMatrix <- confusionMatrix(as.factor(train_predicted_classes), as.factor(y_train))
cat("Training Set Confusion Matrix:\n")
print(train_confusionMatrix)

# 评估训练集上的模型性能
cat("Training Set Classification Report:\n")
print(train_confusionMatrix$byClass)








# 使用最佳模型进行测试集预测，并获取正类的概率
test_predictions <- predict(elastic_net_model, newdata = x_test, type = "prob")[, "Class1"]
test_predicted_classes <- ifelse(test_predictions > 0.5, "Class1", "Class0")

# 生成和评估测试集上的混淆矩阵
test_confusionMatrix <- confusionMatrix(as.factor(test_predicted_classes), as.factor(y_test))
print("Test Set Confusion Matrix:")
print(test_confusionMatrix)

# 评估测试集上的模型性能
print("Test Set Classification Report:")
print(test_confusionMatrix$byClass)

