relu = function(x) pmax(x, 0)
a = -10:10
plot(3*relu(a) ~ a, type='l')
lines(relu(2*a) ~ a, type='l', col='red')
lines(relu(a+2) ~ a, type='l', col='blue')
lines(relu(2*a+2) ~ a, type='l', col='green')


plot(relu(2*a+10) + relu(a) ~ a, type='l', col='black')

plot(relu(2*a+10) + relu(a) + relu(-3*a) + relu(-3*a) ~ a, type='l', col='black')


a = -10:10
plot(2*a+3 ~ a, type='l')
lines(3*a+2 ~ a, type='l', col='red')
lines(2*a+3 + 3*a+2 ~ a, type='l', col='blue')

