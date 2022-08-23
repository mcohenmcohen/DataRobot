rm(list=ls(all=T))
gc(reset=T)
set.seed(42)
n = 1e4
a = rbeta(n,20,.001)*-1
b = rbeta(n,20,.001)
x = c(a, b)

x = (x-mean(x)) / sd(x)
hist(x)
sum(abs(x)>sd(x)) / (n*2)
print(mean(x))
print(sd(x))