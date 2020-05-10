#setwd('/home/chris/Datasets')
setwd('C:\\Users\\rober\\OneDrive\\School\\ISTE780\\Project')

rm(list=ls())

library("devtools")

install_github("davpinto/fastknn")
install.packages('caTools')

library(fastknn)
library(caTools)

df.train <- read.csv('sign_mnist_train.csv', header=TRUE)
df.train_label <- as.factor(df.train$label)
df.train <- as.matrix(df.train)

df.test <-  read.csv('sign_mnist_test.csv', header=TRUE)
df.test_label  <- as.factor(df.test$label)
df.test <- as.matrix(df.test)

## Start Timer
ptm <- proc.time()

## Fit KNN
predicted <- fastknn(df.train, df.train_label, df.test, k = 3)

## Evaluate model on test set
sprintf("Accuracy: %.2f", 100 * (1 - classLoss(actual = df.test_label, predicted = predicted$class)))

# Stop timer
proc.time() - ptm

## [1] "Accuracy: 80.15"
##   user  system elapsed 
## 107.75    0.25  108.43


## 5-fold CV using log-loss as evaluation metric
#set.seed(1738)
#  cv.out <- fastknnCV(df.train, df.train_label, k = 1:10, method = "vote", eval.metric = "logloss", nthread=6)
#cv.out$cv_table


