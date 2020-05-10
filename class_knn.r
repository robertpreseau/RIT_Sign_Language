#setwd('/home/chris/Datasets')
setwd('C:\\Users\\rober\\OneDrive\\School\\ISTE780\\Project')

rm(list=ls())

library(class)

df.train <- read.csv('sign_mnist_train.csv', header=TRUE)
df.test <- read.csv('sign_mnist_test.csv', nrows=100, header=TRUE)
df.train_label <- df.train$label
df.test_label  <- df.test$label

tab <- table(pr,df.test_label)

##this function divides the correct predictions by total number of predictions that tell 
## us how accurate the model is.
##accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
##accuracy(tab)

knn_pred_y = NULL
error_rate_x = NULL

for (x in (1: 20)) {
  # Start the clock!
  ptm <- proc.time()
  
  set.seed(1738)
  knn_pred_y <- knn(df.train,df.test,cl=df.train_label,k=x)
  error_rate_x[x] <- mean(knn_pred_y != df.test_label)
  
  # Stop the clock
  print(proc.time() - ptm)
}

error_rate_x
## 8.943%

k = which(error_rate_x == min(error_rate_x))
print(k)
##3



