library(ggplot2)
library(scales)
data(diamonds)

ggplot(diamonds, aes(clarity, fill=cut)) +
  geom_bar() +
  scale_fill_brewer(type = "qual", palette=2) +
  theme_bw()

ggplot(diamonds, aes(clarity, fill=cut)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  ylab('Percent') +
  scale_fill_brewer(type = "qual", palette=2) +
  theme_bw()
