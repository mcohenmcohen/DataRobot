library(ggplot2)
data(diamonds)
ggplot(diamonds, aes(x=carat, y=price)) + geom_point() + theme_bw()
ggplot(diamonds, aes(x=carat, y=price)) + geom_point(alpha=.05) + theme_bw()
ggplot(diamonds, aes(x=carat, y=price)) + stat_binhex(trans = "log") +
  scale_fill_gradient(name = "count", trans = "log10") +
  theme_bw()
