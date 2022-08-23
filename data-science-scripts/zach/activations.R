

elu = function(x, alpha=1){
  alpha * expm1(x)
}

selu = function(x){
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return(scale * elu(x, alpha))
}


swish = function(x){
  x * plogis(x)
}

y = x
plot(y ~ x, type='l', col='black', lty=3)
lines(elu(x) ~ x, col='black')
lines(selu(x) ~ x, col='blue')
lines(swish(x) ~ x, col='red')

legend(-1, 1, c('elu', 'selu', 'swish'), c('black', 'blue', 'red'))