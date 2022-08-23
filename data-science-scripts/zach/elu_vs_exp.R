##############################
# Mercari
##############################
set.seed(42)
x = rpois(100, .5) + 1
print(c(mean(x), sd(x)))
print(c(mean(log(x)), sd(log(x))))

after_act = exp(log(x) * sd(log(x)) + mean(log(x)))
print(c(mean(after_act), sd(after_act)))
after_scale = after_act*sd(x)+mean(x)
print(c(mean(after_scale), sd(after_scale)))






target = c(rep(0, 90), rep(1, 10))
t_mean = mean(target)
t_sd = sd(target)
target_scaled = (target-t_mean)/t_sd
c(mean(target_scaled), sd(target_scaled))


inv_hard_sigmoid = function(x){
  (x - 0.5) / 0.2
}
target_scaled_inv = inv_hard_sigmoid(target_scaled)
t_mean2 = mean(target_scaled_inv)
t_sd2 = sd(target_scaled_inv)
target_scaled_inv_scaled = (target_scaled_inv-t_mean2) / t_sd2
c(mean(target_scaled_inv_scaled), sd(target_scaled_inv_scaled))

hard_sigmoid = function(x){
  x = (x * 0.2) + 0.5
  x = pmax(x, 0)
  x = pmin(x, 1)
  return(x)
}
out = hard_sigmoid((target_scaled_inv*t_sd2)+t_mean2)


elu = function(x){
  out = x
  out[x<0] <- 1 * (expm1(x[x<0]))
  return(out)
}

inv_elu = function(x){
  out = x
  out[x<0] <- (log1p(x[x<0])) / 1
  return(out)
}

x <- sort(rnorm(1000))
plot(elu(x)~ x, col='blue', type='l')
lines(exp(x) ~ x, col='red')

all.equal(inv_elu(elu(x)), x)

x <- sort(rnorm(1000))
softsign = function(x) {x/(1+abs(x))}
inv_softsign = function(x) {-x / (x-1)}
all.equal(inv_softsign(softsign(x)), x)


selu = function(x){
  out = x
  out[x<0] <- 1.6732 * (expm1(x[x<0]))
  out = 1.0507 * out
  return(out)
}
inv_selu = function(x){
  out = x / 1.0507
  out[x<0] <- (log1p(x[x<0] / (1.6732 * 1.0507)))
  return(out)
}
x <- sort(rnorm(10000))
all.equal(inv_selu(selu(x)), x)
plot(selu(x) ~ x)


relu = function(x){
  pmax(x, 0)
}
inv_relu = function(x){
  out = x
  out[x<=0] <- NA
  return(out)
}
x <- sort(rnorm(10000))
out <- inv_relu(relu(x))
all.equal(out[!is.na(out)], x[!is.na(out)])
plot(inv_relu(relu(x))~x)


hard_sigmoid = function(x){
  x = (x * 0.2) + 0.5
  x = pmax(x, 0)
  x = pmin(x, 1)
  return(x)
}
inv_hard_sigmoid = function(x){
  out = (x - 0.5) / 0.2
  out[x<=0] <- NA
  out[x>=1] <- NA
  return(out)
}
x <- sort(rnorm(10000))
out <- inv_hard_sigmoid(hard_sigmoid(x))
all.equal(out[!is.na(out)], x[!is.na(out)])
plot(inv_hard_sigmoid(hard_sigmoid(x))~x)




swish = function(x){
  return(x/(1+exp(-x)))
}
inv_swish = function(x){
  out = (x - 0.5) / (0.2 * x)
  return(out)
}
x <- sort(rnorm(10000))
plot(swish(x)~x)
out <- inv_swish(swish(x))
all.equal(out[!is.na(out)], x[!is.na(out)])
plot(inv_swish(swish(x))~x)


softsign = function(x){
  return(x/(1+abs(x)))
}
inv_softsign = function(x){
  out1 =- x/(x-1)
  out2 = x/(x+1)
  out = out1
  out[x<0] = out2[x<0]
  return(out)
}
x <- sort(rnorm(10000))
plot(softsign(x)~x)
out <- inv_softsign(softsign(x))
all.equal(out[!is.na(out)], x[!is.na(out)])
plot(inv_softsign(softsign(x))~x)




inv_hard_sigmoid = function(x){
  out = (x - 0.5) / 0.2
  return(out)
}
x=c(rep(0, 4000), rep(1, 6000)); eps=0.0001; y=qlogis(pmax(pmin(x, 1-eps), eps)); plot(y~x); summary(y); sd(y); 
summary(inv_hard_sigmoid(x))
sd(summary(inv_hard_sigmoid(x)))

x=c(rep(0, 1), rep(1, 10000)); eps=0.0001; y=qlogis(pmax(pmin(x, 1-eps), eps)); plot(y~x); summary(y); sd(y); 
summary(inv_hard_sigmoid(x))
sd(summary(inv_hard_sigmoid(x)))


x=c(rep(0, 10000), rep(1, 1)); eps=0.0001; y=qlogis(pmax(pmin(x, 1-eps), eps)); plot(y~x); summary(y); sd(y); 
summary(inv_hard_sigmoid(x))
sd(summary(inv_hard_sigmoid(x)))



leaky_relu = function(x){
  out = x
  idx = x<0
  out[idx] = out[idx] * .01
  return(out)
}

inv_leaky_relu = function(x){
  out = x
  idx = x<0
  out[idx] = out[idx] * 100
  return(out)
}
x <- sort(rnorm(10000))
plot(leaky_relu(x)~x)
out <- inv_leaky_relu(leaky_relu(x))
all.equal(out, x)

