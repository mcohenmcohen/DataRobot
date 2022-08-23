library(data.table)
dat <- fread('~/workspace/data-science-scripts/zach/Historical ownership data - 2017 season.csv')
dat <- dat[!is.na(Ownership),]

player_week <- CJ(
  Player=dat[,sort(unique(Player))],
  Date=dat[,sort(unique(Date))]
)

keys <- c('Player','Date')
dat <- merge(dat, player_week, by=keys, all=T)
setkeyv(dat, keys)

dat[is.na(Ownership) == 0,]

fwrite(dat, '~/workspace/data-science-scripts/zach/Historical ownership data - 2017 season - padded.csv')