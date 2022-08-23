rectangle <- function(x1, y1, x2, y2){
  if(x1 == x2){
    stop('x coordinates must differ')
  }
  if(y1 == y2){
    stop('y coordinates must differ')
  }
  points <- rbind(
    c(x1, y1),
    c(x2, y2),
    c(x1, y2),
    c(x2, y1)
  )
  points <- points[order(points[,1], points[,2]),]
  row.names(points) <- c(
    'point_1',
    'point_2',
    'point_3',
    'point_4'
  )
  out <- list(
    points = points,
    width = max(points[,1]) - min(points[,1]),
    height = max(points[,2]) - min(points[,2])
  )
  out[['area']] <- out[['width']] * out[['height']]
  class(out) <- 'rectangle'
  return(out)
}

print.rectangle <- function(x, ...){
  print(x[['points']])
}

summary.rectangle <- function(object, ...){
  print(paste('Area:', object[['area']]))
}

plot.rectangle <- function(x, y=NULL, label_points=TRUE, ...){
  plot(x[['points']], xlab='x', ylab='y')
  if(label_points){
    text(x[['points']], labels=row.names(x[['points']]))
  }
  lines(x[['points']][c(1,3),])
  lines(x[['points']][c(1,2),])
  lines(x[['points']][c(2,4),])
  lines(x[['points']][c(3,4),])
}

rectangle(1,3,1,3)
rectangle(1,3,3,3)

my.rectangle <- rectangle(1,1,3,3)
print(my.rectangle)
summary(my.rectangle)
plot(my.rectangle)

my.rectangle <- rectangle(3,3,1,1)
print(my.rectangle)
summary(my.rectangle)
plot(my.rectangle)

my.rectangle <- rectangle(2,1,3,2)
print(my.rectangle)
summary(my.rectangle)
plot(my.rectangle)
