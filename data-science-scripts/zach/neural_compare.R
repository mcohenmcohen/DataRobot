############################################################
# Load data
############################################################
require(mxnet)
require(data.table)
train <- fread('~/datasets/digits_train.csv', header=TRUE)
test <- fread('~/datasets/digits_test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]

train.x <- t(train.x/255)
test.x <- t(test/255)

table(train.y)

############################################################
# mxnet
############################################################
#http://mxnet.readthedocs.org/en/latest/R-package/mnistCompetition.html
devices <- lapply(1:8,  mx.cpu)

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

################
#Simple model
################

#Train
mx.set.seed(0)
tic <- proc.time()
model_simple <- mx.model.FeedForward.create(
  softmax, X=train.x, y=train.y,
  ctx=devices, num.round=10, array.batch.size=100,
  learning.rate=0.07, momentum=0.9, eval.metric=mx.metric.accuracy,
  initializer=mx.init.uniform(0.07),
  epoch.end.callback=mx.callback.log.train.metric(100))
mxnet_simple_time <- proc.time() - tic
print(mxnet_simple_time)

#Make preds
preds <- predict(model_simple, test.x)
dim(preds)
pred.label <- max.col(t(preds)) - 1
table(pred.label)

#Make sub - 0.96900
submission <- data.frame(ImageId=1:ncol(test.x), Label=pred.label)
write.csv(
  submission,
  file='~/Desktop/mxnet_simple_submission.csv',
  row.names=FALSE, quote=FALSE)

################
# H20 MLP
################

#Layer factory:
dropout_factory <- function(input, num_hidden, dropout_ratio=0.2){
  fc <- mx.symbol.FullyConnected(input, num_hidden=num_hidden)
  bn = mx.symbol.BatchNorm(data = fc)
  do <- mx.symbol.Dropout(bn, p = dropout_ratio)
  #act <- mx.symbol.LeakyReLU(bn, act.type="rrleu")
  act <- mx.symbol.Activation(bn, act.type="relu")
  return(act)
}

devices <- lapply(1:8,  mx.cpu)

data <- mx.symbol.Variable("data")
l1 <- dropout_factory(data, 1024)
l2 <- dropout_factory(l1, 1024)
l3 <- dropout_factory(l2, 2048)
softmax <- mx.symbol.SoftmaxOutput(l3, name="sm")

#Train
mx.set.seed(0)
tic <- proc.time()
model_mlp <- mx.model.FeedForward.create(
  softmax, X=train.x, y=train.y,
  ctx=devices, num.round=100, array.batch.size=100,
  learning.rate=0.005, momentum=0.01, eval.metric=mx.metric.accuracy,
  initializer=mx.init.uniform(0.07),
  epoch.end.callback=mx.callback.log.train.metric(100))
mxnet_simple_time <- proc.time() - tic
print(mxnet_simple_time)

#Make preds
preds <- predict(model_mlp, test.x)
dim(preds)
pred.label <- max.col(t(preds)) - 1
table(pred.label)

#Make sub - 0.96686
submission <- data.frame(ImageId=1:ncol(test.x), Label=pred.label)
write.csv(
  submission,
  file='~/Desktop/mxnet_h20_mlp_submission.csv',
  row.names=FALSE, quote=FALSE)

################
#Complex model
################
# rehsape data into arrays
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

# Fit model
mx.set.seed(0)
tic <- proc.time()
model_complex <- mx.model.FeedForward.create(
  lenet, X=train.array, y=train.y,
  ctx=devices, num.round=20, array.batch.size=100,
  learning.rate=0.05, momentum=0.9, wd=0.00001,
  eval.metric=mx.metric.accuracy,
  epoch.end.callback=mx.callback.log.train.metric(100))
mxnet_complex_time <- proc.time() - tic
print(mxnet_complex_time)

#Make sub - 0.99086
preds <- predict(model_complex, test.array)
pred.label <- max.col(t(preds)) - 1
submission <- data.frame(ImageId=1:ncol(test.x), Label=pred.label)
write.csv(submission, file='~/Desktop/mxnet_complex_submission.csv', row.names=FALSE, quote=FALSE)

############################################################
# h20
############################################################
#http://tjo-en.hatenablog.com/entry/2015/02/25/125417
#http://h2o.ai/blog/2014/09/r-h2o-domino/

#Setup
library(h2o)
localH2O <- h2o.init(max_mem_size = '12g', nthreads=-1)

train_df <- data.frame(train)
test_df <- data.frame(test)

train_df$label <- factor(train_df$label)

train.hex <- as.h2o(train_df)
test.hex <- as.h2o(test_df)

###########################
#Simple Model
###########################
tic <- proc.time()
h20_simple <- h2o.deeplearning(
  x = 2:785, y = 1, training_frame = train.hex,
  activation = "Tanh",
  hidden=rep(160,5),
  epochs = 20, seed=0)
h20_simple_time <- proc.time() - tic
print(h20_simple_time)

#Submission - 46 seconds and 0.85871 acc trying to use same params as mxnet
#Using tutorial params, 503.397 seconds and 0.96700 acc
pred.dl <- h2o.predict(object=h20_simple, newdata=test.hex)
pred.dl.df <- as.data.frame(pred.dl)
submission <- data.frame(ImageId=1:ncol(test.x), Label=pred.dl.df[,1])
write.csv(submission, file='~/Desktop/h20_simple_submission.csv', row.names=FALSE, quote=FALSE)

###########################
#Complex Model
###########################
tic <- proc.time()
h20_complex <- h2o.deeplearning(
  x = 2:785, y = 1, training_frame = train.hex,
  activation = "RectifierWithDropout", hidden=c(1024,1024,2048),
  epochs = 200, adaptive_rate = FALSE, rate=0.01,
  rate_annealing = 1.0e-6, rate_decay = 1.0, momentum_start = 0.5,
  momentum_ramp = 32000*12, momentum_stable = 0.99, input_dropout_ratio = 0.2,
  l1 = 1.0e-5,l2 = 0.0, max_w2 = 15.0, initial_weight_distribution = "Normal",
  initial_weight_scale = 0.01, nesterov_accelerated_gradient = T,
  loss = "CrossEntropy", fast_mode = T, diagnostics = T, ignore_const_cols = T,
  force_load_balance = T, seed=0)
h20_complex_time <- proc.time() - tic
print(h20_complex_time)

#Predict
pred.dl <- h2o.predict(object=h20_complex, newdata=test.hex)
pred.dl.df <- as.data.frame(pred.dl)
submission <- data.frame(ImageId=1:ncol(test.x), Label=pred.dl.df[,1])
write.csv(submission, file='~/Desktop/h20_complex_submission.csv', row.names=FALSE, quote=FALSE)


#Cpu times
#mxnet with 4 CPUs
#h20 with 8 threads
mxnet - 3 layer MLP - 14 seconds - 97% accuracy
mxnet - 4 layer covnet - 150 seconds - 98% accuracy
h20 - 2 layer MLP - 500 seconds - 97% accuracy
h20 - 3 layer MLP - still running after an hour


