#setwd('/home/chris/Datasets')
setwd('C:\\Users\\rober\\OneDrive\\School\\ISTE780\\Project')

rm(list=ls())
install.packages("beepr")
library(class)
library(beepr)

df.train <- read.csv('sign_mnist_train.csv', header=TRUE)
df.test <- read.csv('sign_mnist_test.csv', nrows=100, header=TRUE)

df.train_label <- df.train$label
df.test_label  <- df.test$label

df.train <- df.train[,-1]
df.test <- df.test[,-1]

linearMod <- lm(df.train_label ~ ., data=df.train)  # build linear regression model on full data
print(linearMod)
summary(linearMod)
AIC(linearMod)
BIC(linearMod)

library(MASS)
step <- stepAIC(linearMod, direction = "backward", trace = FALSE)
step

beep(4)
