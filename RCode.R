setwd('/Users/tomo/Desktop/7152/Ford_Alert')
getwd()

data <- read.csv(file = "fordTrain.csv")
names(data)
dim(data)

newdata <- data[,3:ncol(data)]
dim(newdata)
names(newdata)

smp_size <- floor(0.70 * nrow(newdata))
## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(newdata)), size = smp_size)
training <- newdata[train_ind, ]
testing <- newdata[-train_ind, ]

#---------------------------------------------------------#
#L1 Logistic Regression - glmnet-super slow
#---------------------------------------------------------#
library(glmnet)
training <- as.matrix(na.omit(training))
testing <- as.matrix(na.omit(testing))
fit <- glmnet(training[,2:ncol(training)], training[,1], family="binomial")
plot(fit)
cv3 <- cv.glmnet(training[,2:ncol(training)], training[,1], family="binomial", type="class")
plot(cv3)
cv3$lambda.min
cv3$lambda.1se
pred2 <- as.vector(predict(fit, testing[,2:ncol(testing)], type="class", s=cv3$lambda.min))
pred2 <- data.frame(pred2)
testing <- data.frame(testing)
glmnet.table <- table(pred2[,1], testing$IsAlert)
1-sum(diag(glmnet.table))/sum(glmnet.table)

library(pROC)
roc.curve <- roc(as.numeric(pred2[,1])-1, testing$IsAlert)
plot(roc.curve, main = "ROC: Logistic Regression", col = "red")
auc.score<-auc(testing$IsAlert, as.numeric(pred2[,1])-1)
auc.score
#[1] 0.7803548

#---------------------------------------------------------#
#classification tree (0, 1 tree-model) fast
#---------------------------------------------------------#
library(tree)
training$IsAlert <- factor(training$IsAlert)
tree.fit <- tree(IsAlert~.,data=training)
summary(tree.fit)
plot(tree.fit)
text(tree.fit, pretty=0)
tree.fit

set.seed(2)
testing$IsAlert <- as.factor(testing$IsAlert)
tree.pred <- predict(tree.fit, testing, type="class")
tree.table <- table(tree.pred, testing$IsAlert)

library(pROC)
roc.curve <- roc(as.numeric(tree.pred)-1, as.numeric(testing$IsAlert)-1)
plot(roc.curve, main = "ROC: Classification Tree", col = "red")
auc.score<-auc(as.numeric(testing$IsAlert)-1, as.numeric(tree.pred)-1)
auc.score
#[1] 0.8287284

#---------------------------------------------------------#
#LR - GLM
#---------------------------------------------------------#
training <- data.frame(training)
training <- na.omit(training)
lr.fit <- glm(IsAlert~., data=training)
summary(lr.fit)
#P8, V7 and V9 are redundant/useless variables

testing <- data.frame(testing)
testing <- na.omit(testing)
lr.pred <- predict(lr.fit, testing[,-1], type="response")

lr.pred[lr.pred>=0.5] <- 1
lr.pred[lr.pred<0.5] <- 0
lr.table <- table(lr.pred, testing$IsAlert)
1-sum(diag(lr.table))/sum(lr.table) #misclassification rate

library(pROC)
set.seed(100)
roc.curve <- roc(as.numeric(lr.pred), testing$IsAlert)
plot(roc.curve, main = "ROC: GLM", col = "red")
auc.score<-auc(as.numeric(testing$IsAlert), as.numeric(lr.pred))
auc.score
#[1] 0.741541

#exclude useless variables
lr.fit <- glm(IsAlert~.-P8-V7-V9-P2-E4-E7-E11, data=training)

#---------------------------------------------------------#
#RF - good AUC score
#---------------------------------------------------------#
library(randomForest)
set.seed(100)
#500 ntree
RF <- randomForest(training[,-c(1,9,27,29)], factor(training$IsAlert),
                   sampsize=10000, do.trace=TRUE, importance=TRUE, ntree=10, forest=TRUE)
varImpPlot(RF)
rf.pred <- data.frame(IsAlert.pred=predict(RF,testing[,-c(1,9,27,29)],type="prob")[,2])

library(pROC)
set.seed(10)
roc.curve <- roc(rf.pred[,1], as.numeric(testing$IsAlert))
plot(roc.curve, main = "ROC: RF", col = "red")
auc.score<-auc(as.numeric(testing$IsAlert), rf.pred[,1])
auc.score

rf.pred[rf.pred>=0.5] <- 1
rf.pred[rf.pred<0.5] <- 0
rf.table <- table(pred=rf.pred[,1], testing$IsAlert)
1-sum(diag(rf.table))/sum(rf.table) #misclassification rate

#---------------------------------------------------------#
#Naive Bayes(gaussian) - much faster than SVM 
#---------------------------------------------------------#
#http://stackoverflow.com/questions/20091614/naive-bayes-classifier-in-r
## Categorical data only
library(e1071)
# training$IsAlert[training$IsAlert==1] <- 'TRUE'
# training$IsAlert[training$IsAlert==0] <- 'FALSE'
# testing$IsAlert[testing$IsAlert==1] <-  'TRUE'
# testing$IsAlert[testing$IsAlert==0] <- 'FALSE'
training <- na.omit(training)
testing <- na.omit(testing)
training <- data.frame(training)
testing <- data.frame(testing)
training$IsAlert <- as.factor(training$IsAlert)
testing$IsAlert <- as.factor(testing$IsAlert)
model <- naiveBayes(IsAlert~., data=training)
NB.pred <- predict(model, testing)
NB.table <- table(NB.pred, testing$IsAlert)
1-sum(diag(NB.table))/sum(NB.table)

library(pROC)
roc.curve <- roc(as.numeric(NB.pred)-1, as.numeric(testing$IsAlert)-1)
plot(roc.curve, main = "ROC: Naive Bayes", col = "red")
auc.score<-auc(as.numeric(testing$IsAlert)-1, as.numeric(NB.pred)-1)
auc.score
#[1] 0.6774233

#---------------------------------------------------------#
#CART - regression tree
#---------------------------------------------------------#
library(rpart) #grow a regression tree
set.seed(1)
training <- data.frame(training)
rpart.fit <- rpart(IsAlert~., data=training, control=rpart.control(minsplit = 10))
par(xpd=NA)
plot(rpart.fit, uniform = T)
text(rpart.fit, use.n = TRUE)

set.seed(2)
testing$IsAlert <- as.factor(testing$IsAlert)
rpart.pred <- predict(rpart.fit, testing, type="class")
rpart.table <- table(rpart.pred, testing$IsAlert)
1-sum(diag(rpart.table))/sum(rpart.table)

library(pROC)
roc.curve <- roc(as.integer(rpart.pred)-1, as.numeric(testing$IsAlert)-1)
plot(roc.curve, main = "Logistic Regression ROC Curve", col = "red")
auc.score<-auc(as.numeric(testing$IsAlert)-1, as.numeric(rpart.pred)-1)
auc.score

library("partykit")
plot(as.party(rpart.fit), tp_args = list(id=FALSE))
print(rpart.fit$cptable)
opt <- which.min(rpart.fit$cptable[,"xerror"])
cp <- rpart.fit$cptable[opt, "CP"]
rpart_prune <- prune(rpart.fit, cp=cp)
plot(as.party(rpart_prune), tp_args = list(id=FALSE))

# pred <- predict(rpart_prune, newdata=testing)
# xlim <- range(as.numeric(testing$IsAlert)-1)
# plot(pred~IsAlert, data=testing, xlab="Observed", ylab="Predicted", ylim=xlim, xlim=xlim)
# abline(a=0,b=1)

