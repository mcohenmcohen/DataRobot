#Setup
library(data.table)
library(pbapply)
library(quantmod)
library(TTR)

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

#Adjust data and calculate co-variates
pp_data <- function(a){
  x <- adjustOHLC(a, use.Adjusted=TRUE)
  x <- to.monthly(x)
  colnames(x) <- c('Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted')

  C <- log(x[,'Close'])
  HLC <- log(x[,c('High', 'Low', 'Close')])
  HL <- log(x[,c('High','Low')])
  V <- log(x[,'Volume'])

  #Diff signals that need to be compared to the raw prices
  alma <- ALMA(C, n = 3)
  colnames(alma) <- 'alma'
  bbands <- BBands(HLC, n = 3)
  diffs <- cbind(
    sema = SMA(C, n = 4),
    ema = EMA(C, n = 4),
    dema = DEMA(C, n = 3),
    wma = WMA(C, n = 4),
    evwma = EVWMA(C, V, n = 4),
    zlema = ZLEMA(C, n = 4),
    vwap = VWAP(C, V, n = 4),
    hma = HMA(C, n = 4),
    alma = alma
  )
  diffs <- lapply(diffs, function(b) b - C)
  diffs <- do.call(cbind, diffs)

  #Normalized signals
  adx <- ADX(HLC, n=3)
  adx <- (adx$DIp - adx$DIn) / adx$DX
  names(adx) <- 'adx'

  dpo <- DPO(C, shift=0, n=3)
  names(dpo) <- 'dpo'

  mom <- momentum(C)
  names(mom) <- 'mom'

  cmo <- CMO(C, n=3)
  names(cmo) <- 'cmo'

  rsi <- RSI(C, n=3, wts=V)
  names(rsi) <- 'rsi'

  stochRSI <- stoch(rsi, nFastK = 4, nFastD = 1, nSlowD = 1)$fastK
  names(stochRSI) <- 'stochRSI'

  wpr <- WPR(HLC, n=3)
  names(wpr) <- 'wpr'

  cmf <- lapply(1:6, function(x){
    cmf <- CMF(HLC, V, n=x)
    names(cmf) <- paste0('cmf', x)
    return(cmf)
  })
  cmf <- Reduce(cbind, cmf)

  signals <- cbind(
    adx,
    ATR(HLC, n=3)[,c('tr', 'atr')],
    bb_pct = bbands$pctB,
    CCI(HLC, n=6),
    cmo,
    dpo,
    mom,
    SMI(HLC, nFast=1, nSlow=3, nSig=4),
    stochRSI,
    wpr,
    cmf,
    CLV(HLC)
  )

  #Lags
  DIFF <- cbind(diff(C, 1), diff(C, 2), diff(C, 3))
  names(DIFF) <- paste0('DIFF', 1:3)

  #Assemble it all
  mat <- cbind(
    DIFF,
    signals,
    diffs
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
  dat[year >= 2013, set := 'valid']
  dat[year >= 2015, set := 'holdout']

  #Remove NAs
  no_nas <- which(!apply(dat, 1, anyNA))
  dat <- dat[no_nas,]

  #Name and return
  name <- strsplit(names(a)[1], '.', fixed=TRUE)[[1]][1]
  dat[,symbol := name]
  return(dat)
}
data_pp_list <- lapply(data_list, pp_data)

#Combine and save
dat <- rbindlist(data_pp_list)
dat[,table(symbol, set)]
dat[,table(year, set)]
dat[,date := NULL]
write.csv(dat, '~/datasets/etfs.csv', row.names=FALSE)
