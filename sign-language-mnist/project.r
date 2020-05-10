setwd('C:\\Users\\rober\\OneDrive\\School\\ISTE780\\Project\\sign-language-mnist')

rm(list=ls()) 

#tensorflow::install_tensorflow(method=c("conda"), conda="auto", envname="r-tensorflow")
install.packages('ggplot2')
install.packages('readr')
install.packages('caret')
install.packages('data.table')


# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 
set.seed(1738)

library('devtools')
library('tensorflow')
library('keras')
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(caret) # mark up training and test set categoricals
library(keras) # interface to tensorflow
library(data.table)

# see examples install instructions here
# https://www.r-bloggers.com/keras-for-r/amp/
# https://cran.r-project.org/web/packages/keras/vignettes/faq.html
# if CPU install of tensorflow then just install with gpu
# these likely will best run locally not in kernel
# install_keras() 
#install_keras(tensorflow = "gpu")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


train_file <- "sign_mnist_train.csv"
test_file  <- "sign_mnist_test.csv"
category   <- c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25")

fmnist  <- read.csv(train_file)
fmnist2 <- read.csv(test_file)

#mnist <- dataset_mnist()
train_data <- data.matrix(fmnist[,-1]) # strip labels
train_label <- fmnist$label

df<-data.table(fmnist$label)

colnames(df)<-c("X")
train_labels<-df[X == 0, Label := "A"]
train_labels<-df[X == 1, Label := "B"]
train_labels<-df[X == 2, Label := "C"]
train_labels<-df[X == 3, Label := "D"]
train_labels<-df[X == 4, Label := "E"]
train_labels<-df[X == 5, Label := "F"]
train_labels<-df[X == 6, Label := "G"]
train_labels<-df[X == 7, Label := "H"]
train_labels<-df[X == 8, Label := "I"]
train_labels<-df[X == 9, Label := "J"]
train_labels<-df[X == 10, Label := "K"]
train_labels<-df[X == 11, Label := "L"]
train_labels<-df[X == 12, Label := "M"]
train_labels<-df[X == 13, Label := "N"]
train_labels<-df[X == 14, Label := "O"]
train_labels<-df[X == 15, Label := "P"]
train_labels<-df[X == 16, Label := "Q"]
train_labels<-df[X == 17, Label := "R"]
train_labels<-df[X == 18, Label := "S"]
train_labels<-df[X == 19, Label := "T"]
train_labels<-df[X == 20, Label := "U"]
train_labels<-df[X == 21, Label := "V"]
train_labels<-df[X == 22, Label := "W"]
train_labels<-df[X == 23, Label := "X"]
train_labels<-df[X == 24, Label := "Y"]
train_labels<-df[X == 25, Label := "Z"]

test_data <- data.matrix(fmnist2[,-1])
test_label <- fmnist2$label

#one.hot.labels <- decodeClassLabels(train_label)
#one.hot.labels2 <- decodeClassLabels(test_label)

train_data_process <- train_data
train_data_label <- train_label

test_data_process  <- test_data
test_data_label  <- test_label



# reshape
dim(train_data_process) <- c(nrow(train_data_process), 784)
dim(test_data_process)  <- c(nrow(test_data_process), 784)

# rescale
train_data_process <- train_data_process / 255
test_data_process  <- test_data_process / 255

# The y data is an integer vector with values ranging from 0 to 25. 
# To prepare this data for training we one-hot encode the vectors 
# into binary class matrices using the Keras to_categorical() function:
train_data_label <- to_categorical(train_data_label, 25)
test_data_label  <- to_categorical(test_data_label, 25)

# defined labels
#colnames(train_data_label) = c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24")
colnames(train_data_label)=c("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y")
#colnames(test_data_label) = c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25")
colnames(test_data_label)=c("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y")


#----------------------#
# visualizations
# see individual images
show_letter <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}
show_letter2 <- function(arr784, col=gray(255:1/255), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# show a "D"
train_labels$Label[9]
show_letter2(train_data[9,]) 

# show a "N"
train_labels$Label[5]
show_letter(train_data[5,]) 

# show a "W"
train_labels$Label[8] 
show_letter2(train_data[8,])

# begin layering some models
# for this demo, show the following
# 1. linear dense layers with dropout
# 2. simple convnet with max pooling for translations
# 3. variations on convnet architecture
# 4. recurrent neural net on sequential pixels
# 5. Long-term, short-term memory (LSTM)

# 1. linear stack of layers
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 25, activation = 'softmax')


summary(model)
# compile model with loss, optimizer and metrics defined
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
) 

# training
history <- model %>% fit(
  train_data_process, train_data_label, 
  epochs = 200, 
  batch_size = 128, 
  validation_split = 0.2
)

# testing
model %>% evaluate(test_data_process, test_data_label,verbose = 0)

#loss  0.3574352
#acc   0.8852

# predict
model %>% predict_classes(test_data_process)


#show the final confusion matrix 
table(test_label, predict_classes(model, test_data_process))




#-------------------------#
# 2. Trains a simple convnet on the MNIST dataset
# input image dimensions
img_rows <- 28
img_cols <- 28
num_classes <- 25

dim(train_data_process) <- c(nrow(train_data_process), img_rows, img_cols, 1) 
dim(test_data_process)  <- c(nrow(test_data_process), img_rows, img_cols, 1)
input_shape  <- c(img_rows, img_cols, 1)

cat('train_data_process_shape:', dim(train_data_process), '\n')
cat(nrow(train_data_process), 'train samples\n')
cat(nrow(test_data_process), 'test samples\n')

# define model
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', 
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')



summary(model)
# compile model with loss, optimizer and metrics defined
# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# training
history <- model %>% fit(
  train_data_process, train_data_label, 
  epochs = 2,  # 150
  batch_size = 32, # 128
  validation_split = 0.2,
  verbose = 1
)
# testing
model %>% evaluate(test_data_process, test_data_label,verbose = 0)
scores <- model %>% evaluate(
  test_data_process, test_data_label, verbose = 0
)

cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')
# Test loss: 0.2083899 
# Test accuracy: 0.9293 50 epoch
#
# Test accuracy: 0.9368 150 epoch
# Test loss: 0.2385474

# Test accuracy: 0.9292 10 epoch
# Test loss: 0.219011
# predict
model %>% predict_classes(test_data_process)
#



#-------------------------------#
# 3. change the Convnet architecture
# define model
# https://www.kaggle.com/bugraokcu/cnn-with-keras
# input image dimensions
img_rows <- 28
img_cols <- 28
num_classes <- 25
dim(train_data_process) <- c(nrow(train_data_process), img_rows, img_cols, 1) 
dim(test_data_process) <- c(nrow(test_data_process), img_rows, img_cols, 1)
input_shape <- c(img_rows, img_cols, 1)
cat('train_data_process_shape:', dim(train_data_process), '\n')
cat(nrow(train_data_process), 'train samples\n')
cat(nrow(test_data_process), 'test samples\n')


w <- 3 # moving window as 3x3 or 4x4
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(w,w), activation = 'relu', 
                kernel_initializer='he_normal',
                input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(w,w),  activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(w,w), activation = 'relu') %>%
  layer_dropout(rate = 0.40) %>% 
  # begin fully connected layers
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = num_classes, activation = 'softmax')



summary(model)
# compile model with loss, optimizer and metrics defined
# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# training
history <- model %>% fit(
  train_data_process, train_data_label, 
  epochs = 15, 
  batch_size = 128, 
  validation_split = 0.2,
  verbose = 1
)
# testing
model %>% evaluate(test_data_process, test_data_label,verbose = 0)
scores <- model %>% evaluate(
  test_data_process, test_data_label, verbose = 0
)

cat('Test loss:', scores[[1]], '\n')
# Test loss: 0.2070925
cat('Test accuracy:', scores[[2]], '\n')
# Test accuracy: 0.9286
save_model_hdf5(model, 'my_model.h5')
model <- load_model_hdf5('my_model.h5')

#-------------------------------#
# change the architecture again
# define model
# https://www.kaggle.com/bugraokcu/cnn-with-keras
# input image dimensions
img_rows <- 28
img_cols <- 28
num_classes <- 25
dim(train_data_process) <- c(nrow(train_data_process), img_rows, img_cols, 1) 
dim(test_data_process) <- c(nrow(test_data_process), img_rows, img_cols, 1)
input_shape <- c(img_rows, img_cols, 1)
cat('train_data_process_shape:', dim(train_data_process), '\n')
cat(nrow(train_data_process), 'train samples\n')
cat(nrow(test_data_process), 'test samples\n')


w <- 3 # moving window as 3x3 or 4x4
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(w,w), activation = 'relu', 
                kernel_initializer='he_normal',
                input_shape = input_shape) %>% 
  #  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(w,w),  activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  #  layer_conv_2d(filters = 128, kernel_size = c(w,w), activation = 'relu') %>%
  #  layer_dropout(rate = 0.40) %>% 
  #  layer_conv_2d(filters = 128, kernel_size = c(w,w), activation = 'relu') %>%
  #  layer_dropout(rate = 0.30) %>% 
  # begin fully connected layers
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')



summary(model)
# compile model with loss, optimizer and metrics defined
# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
# if validation loss isn't decreasing anymore
early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 2)

# training
history <- model %>% fit(
  train_data_process, train_data_label, 
  epochs = 250, 
  batch_size = 128, 
  verbose = 1,
  shuffle=TRUE,
  #  callbacks = c(early_stopping),
  #  validation_data=c(test_data_process, test_data_label)
  validation_split = 0.2
  
)
# testing
model %>% evaluate(test_data_process, test_data_label,verbose = 0)
scores <- model %>% evaluate(
  test_data_process, test_data_label, verbose = 0
)

cat('Test loss:', scores[[1]], '\n')
# Test loss: 0.2070925
cat('Test accuracy:', scores[[2]], '\n')
# Test accuracy: 0.9286
save_model_hdf5(model, 'my_model.h5')
model <- load_model_hdf5('my_model.h5')


#-----------------------#
# 4. Recurrent Neural Net
# pixel-by-pixel sequential MNIST in
# A Simple Way to Initialize Recurrent Networks of Rectified Linear Units
# try first on handwriting
# this requires a 3 dim input_size unlike Conv
batch_size <- 32
num_classes <- 25
epochs <- 50
hidden_units <- 100
learning_rate <- 1e-6
clip_norm <- 1.0
# input image dimensions
img_rows <- 28
img_cols <- 28
num_classes <- 25
dim(train_data_process) <- c(nrow(train_data_process), img_rows, img_cols) 
dim(test_data_process) <- c(nrow(test_data_process), img_rows, img_cols)
input_shape <- c(img_rows, img_cols)
batch_input_shape <- c(batch_size,img_rows, img_cols)
cat('train_data_process_shape:', dim(train_data_process), '\n')
cat(nrow(train_data_process), 'train samples\n')
cat(nrow(test_data_process), 'test samples\n')
cat("Evaluate IRNN...\n")
model <- keras_model_sequential()
model %>%  layer_simple_rnn(units = hidden_units,
                            kernel_initializer = initializer_random_normal(stddev = 0.01),
                            recurrent_initializer = initializer_identity(gain = 1.0),
                            #                   recurrent_dropout=0.3,
                            #                   stateful=TRUE,
                            #                   shuffle=FALSE,
                            activation = 'relu',
                            #                   batch_input_shape =  batch_input_shape,
                            input_shape = input_shape) %>% 
  layer_dense(units = num_classes) %>% 
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = learning_rate),
  metrics = c('accuracy')
)
# if validation loss isn't decreasing anymore
early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 3)

history <- model %>% fit(
  train_data_process, train_data_label,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  callbacks = c(early_stopping),
  validation_split = 0.2
  #  validation_data = list(test_data_process, test_data_label)
)

scores <- model %>% evaluate(test_data_process, test_data_label, verbose = 0)
cat('IRNN test score:', scores[[1]], '\n')
cat('IRNN test accuracy:', scores[[2]], '\n')
# Test accuracy: 0.9286
save_model_hdf5(model, 'my_rnn_model.h5')
#model <- load_model_hdf5('my_model.h5')


#-----------------------#
# 5. LSTM 
# long-term short-term memory units
batch_size <- 100
num_classes <- 25
epochs <-250
hidden_units <- 128
learning_rate <- 1e-3
clip_norm <- 1.0
# input image dimensions
img_rows <- 28
img_cols <- 28
num_classes <- 25
dim(train_data_process) <- c(nrow(train_data_process), img_rows, img_cols) 
dim(test_data_process) <- c(nrow(test_data_process), img_rows, img_cols)
input_shape <- c(img_rows, img_cols)
batch_input_shape <- c(batch_size,img_rows, img_cols)
cat('train_data_process_shape:', dim(train_data_process), '\n')
cat(nrow(train_data_process), 'train samples\n')
cat(nrow(test_data_process), 'test samples\n')
cat("Evaluate LSTM...\n")
model <- keras_model_sequential()
model %>%  layer_lstm(units = hidden_units,
                      kernel_initializer = "glorot_uniform",
                      recurrent_initializer = "orthogonal", 
                      recurrent_activation = "hard_sigmoid",
                      bias_initializer="zeros",
                      unit_forget_bias = TRUE,
                      #                   recurrent_dropout=0.3,
                      #                   stateful=TRUE,
                      #                   shuffle=FALSE,
                      activation = 'tanh',
                      #                   batch_input_shape =  batch_input_shape,
                      input_shape = input_shape) %>% 
  layer_dense(units = num_classes) %>% 
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = learning_rate),
  metrics = c('accuracy')
)
# if validation loss isn't decreasing anymore
early_stopping <- callback_early_stopping(monitor = 'val_loss', patience =15)
# history <-
model %>% fit(
  train_data_process, train_data_label,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  callbacks = c(early_stopping),
  validation_split = 0.2
  #  validation_data = list(test_data_process, test_data_label)
)

scores <- model %>% evaluate(test_data_process, test_data_label, verbose = 0)
cat('LSTM test score:', scores[[1]], '\n')
cat('LSTM test accuracy:', scores[[2]], '\n')
# Test accuracy: 0.9286
save_model_hdf5(model, 'my_lstm_model.h5')
#model <- load_model_hdf5('my_model.h5')
summary.keras.engine.training.Model(model)

# Any results you write to the current directory are saved as output.










