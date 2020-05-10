rm(list=ls())

wd = '/users/robertpreseau/OneDrive/School/ISTE780/Project'
setwd(wd)

df.train <- read.csv('sign_mnist_train.csv')
df.test <- read.csv('sign_mnist_test.csv')

nrow(df.train)
#27455
nrow(na.omit(df.train))
#27455

nrow(df.test)
#7172
nrow(na.omit(df.test))
#7172

##table() nicely sums up the totals for us
df.train$label


#visualize
install.packages('ggplot2')
library(ggplot2)

barplot(table(df.train$label), xlab="Character", ylab="Count", xlim=400)

ggplot(as.data.frame(table(df.train$label)), aes(factor(Depth), Freq, fill = Species)) +     
  geom_col(position = 'dodge')


##table() nicely sums up the totals for us
sums_by_label.test <- table(df.test$label)
sums_by_label.test

#let's visualize the data
barplot(sums_by_label.test)


df.train[4,]$label

to_test <- df.train[4,]
img_raw <-matrix((to_test[1,2:ncol(df.train)]), nrow=28, ncol=28)
img_num <- apply(img_raw, 2, as.numeric)
img_num <- apply(img_num, 1, rev)
image(1:28, 1:28, t(img_num), col=gray((0:255)/255))

