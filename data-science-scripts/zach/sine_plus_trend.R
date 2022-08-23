library(data.table)
x=1:1000
y = 1000 + 2*x + 250*sin(x/10)
y = y + ifelse(x>500, 10*x-10*500, 0)
plot(y~x, type='l')
fwrite(data.table(y, x), '~/datasets/sine_plus_trend.csv')