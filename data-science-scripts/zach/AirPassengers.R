library(forecast)
data("AirPassengers")
x <- data.frame(
  date = seq.Date(from=as.Date('1949-01-01'), by='months', length.out=length(AirPassengers)),
  AirPassengers = AirPassengers
)
write.csv(x, '~/datasets/AirPassengers.csv', row.names=FALSE)
system('head ~/datasets/AirPassengers.csv')
