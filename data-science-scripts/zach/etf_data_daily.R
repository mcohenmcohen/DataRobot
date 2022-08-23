#Setup
library(data.table)
library(pbapply)
library(quantmod)
library(TTR)
library(matrixStats)
library(RcppArmadillo)
rm(list=ls(all=TRUE))
gc(reset=TRUE)

#Choose funds
symbol_list <- list(
  list('BND', 'Vanguard Total Bond Market Index Fund'),
  list('VTI', 'Vanguard Total Stock Market Index Fund'),
  list('VXUS', 'Vanguard Total International Stock Index Fund'),
  list('VNQ', 'Vanguard REIT Index Fund'),
  list('VEA', 'Vanguard FTSE Developed Markets ETF'),
  list('VWO', 'Vanguard MSCI Emerging Markets ETF'),
  list('VNQI', 'Vanguard Global ex-US Real Estate ETF'),
  list('VOO', 'Vanguard S&P 500 Etf'),
  list('IVOO', 'Vanguard S&P Mid Cap 400 Index ETF'),
  list('VIOO',  'Vanguard High Dividend Yield ETF'),
  list('VSS', 'Vanguard FTSE All-World ex-US Small Cap Index ETF'),
  list('VYM', 'Vanguard High Dividend Yield ETF')
)
symbol_list <- rbindlist(lapply(symbol_list, as.data.table))
setnames(symbol_list, c('symbol', 'Name'))

#Load data from yahoo finance
data_list <- pblapply(symbol_list$symbol, getSymbols, auto.assign=FALSE)
a <- data_list[[9]]

#Adjust data and calculate co-variates
pp_data <- function(a){
  x <- adjustOHLC(a, use.Adjusted=TRUE)
  #x <- to.monthly(x)
  colnames(x) <- c('Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted')

  C <- log(x[,'Close'])
  HLC <- log(x[,c('High', 'Low', 'Close')])
  HL <- log(x[,c('High','Low')])
  V <- x[,'Volume']

  #Diff signals that need to be compared to the raw prices
  alma <- ALMA(C)
  colnames(alma) <- 'alma'
  bbands <- BBands(HLC)
  diffs <- cbind(
    sema = SMA(C),
    ema = EMA(C),
    dema = DEMA(C),
    wma = WMA(C),
    evwma = EVWMA(C, V),
    zlema = ZLEMA(C),
    vwap = VWAP(C, V),
    hma = HMA(C),
    alma = alma
  )
  diffs <- lapply(diffs, function(b) b - C)
  diffs <- do.call(cbind, diffs)

  #Normalized signals
  adx <- ADX(HLC)
  adx <- (adx$DIp - adx$DIn) / adx$DX
  names(adx) <- 'adx'

  dpo <- DPO(C, shift=0)
  names(dpo) <- 'dpo'

  mom <- momentum(C)
  names(mom) <- 'mom'

  cmo <- CMO(C)
  names(cmo) <- 'cmo'

  rsi <- RSI(C, wts=V)
  names(rsi) <- 'rsi'

  stochRSI <- stoch(rsi)$fastK
  names(stochRSI) <- 'stochRSI'

  wpr <- WPR(HLC)
  names(wpr) <- 'wpr'

  cmf <- lapply(1:7, function(x){
    cmf <- CMF(HLC, V, n=x)
    names(cmf) <- paste0('cmf', x)
    return(cmf)
  })
  cmf <- Reduce(cbind, cmf)

  signals <- cbind(
    adx,
    ATR(HLC)[,c('tr', 'atr')],
    bb_pct = bbands$pctB,
    CCI(C),
    cmo,
    dpo,
    mom,
    SMI(HLC),
    stochRSI,
    wpr,
    cmf,
    CLV(HLC)
  )

  #Lags
  maxdiff <- 14
  DIFF <- lapply(1:maxdiff, function(x){diff(C, x)})
  DIFF <- Reduce(cbind, DIFF)
  names(DIFF) <- paste0('DIFF', 1:maxdiff)

  #Rolling regression
  linear_results <- function(n=14){
    x <- cbind(rep(1, n), 1:n)
    xp <- cbind(1, n+1)
    quick_lm <- function(y){
      mod <- RcppArmadillo::fastLm(x, y)
      cf <- coef(mod)
      p <- predict(RcppArmadillo::fastLm(x, y), xp)
      return(c(cf, p))
    }
    linear_pred <- apply(DIFF[,1:n], 1, quick_lm)
    linear_pred <- t(linear_pred)
    colnames(linear_pred) <- c('linear_pred', 'linear_intercept', 'linear_cf')
    sds <- rowSds(DIFF)
    linear_pred_adj <- linear_pred[,1] / sds
    out <- cbind(
      linear_pred,
      linear_pred_adj,
      sds
    )
    colnames(out) <- paste0(colnames(out), '_', n)
    out <- xts(out, order.by=index(DIFF))
    return(out)
  }
  lin7 <- linear_results(7)
  lin14 <- linear_results(14)

  #Assemble it all
  mat <- cbind(
    DIFF,
    signals,
    diffs,
    lin7,
    lin14
  )
  dat <- data.table(
    date = as.POSIXct(index(mat)),
    year = year(as.POSIXct(index(mat))),
    symbol='UNK',
    target=as.numeric(NA),
    mat
  )
  dat[,target := c(na.omit(DIFF1), NA)]
  #print(dat[,table(year, set)])
  #print(sort(apply(dat, 2, function(x) sum(is.na(x)))))

  #Determine sets
  dat[, set := 'train']
  dat[year >= 2015, set := 'valid']
  dat[year >= 2016, set := 'holdout']

  #Remove NAs
  no_nas <- which(!apply(dat, 1, anyNA))
  dat <- dat[no_nas,]

  #Name and return
  name <- strsplit(names(a)[1], '.', fixed=TRUE)[[1]][1]
  dat[,symbol := name]
  return(dat)
}
data_pp_list <- pblapply(data_list, pp_data)

#Combine and save
dat <- rbindlist(data_pp_list)
dat[,table(symbol, set)]
dat[,table(year, set)]
dat[, wday := paste0('w', wday(date))]
dat[, mday := paste0('m', mday(date))]
dat[, month := paste0('m', month(date))]
dat[, year := paste0('y', year(date))]
dat[,date := NULL]
#dat[,symbol := NULL]
dat[,round(mean(target), 3), by='symbol'][,max(abs(V1))]
dat[,round(median(target), 3), by='symbol'][,max(abs(V1))]
write.csv(dat, '~/datasets/etfs_daily.csv', row.names=FALSE)
