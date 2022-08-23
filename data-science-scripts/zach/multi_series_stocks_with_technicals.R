# Setup
rm(list = ls(all=T))
gc(reset=T)
library(rvest)
library(quantmod)
library(pbapply)
library(data.table)

# Load index
url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500 = html_table(html_nodes(read_html(url), "table")[[1]], fill=T)
tickers = sp500[['Ticker symbol']]
tickers = setdiff(tickers, c('LMT', 'NWL', 'NBL'))

# Load stock data
# Google missing end of 2003
getSymbols(tickers, src="google", from="2004-01-01", auto.assign=TRUE)

# Impute missing and adjust for splits/dividends
tickers_adj = pblapply(tickers, function(x){
  dat = get(x)
  names(dat) <- gsub(paste0('^', x), '', names(dat))
  names(dat) <- gsub('.', '', names(dat), fixed=T)
  if(anyNA(dat)){
    # print(paste(x, 'has missings'))
    dat <- na.locf(dat, na.rm=FALSE)
  }
  if(! x %in% c('BRK.B', 'BF.B')){
    dat = adjustOHLC(dat, 'dividend', symbol.name=x) # Google already split adjusted)
  }
  if(nrow(dat)>365){
    return(dat)
  }
  else{
    return(NULL)
  }

})
names(tickers_adj) = tickers

# Add technicals
tickers_adj <- tickers_adj[!sapply(tickers_adj, is.null)]
tickers_tech = pblapply(names(tickers_adj), function(n){
  print(n)
  adj_ohlc_dat = tickers_adj[[n]]

  # Fix volume
  vol = as.numeric(adj_ohlc_dat[,'Volume'])
  vol[vol==0] <- NA
  vol <- na.approx(vol, na.rm=FALSE)
  vol <- na.locf(vol, na.rm=FALSE)
  vol[is.na(vol)] <- 0
  adj_ohlc_dat[,'Volume'] <- vol

  # Target and some simple metrics
  ClCL <- diff(log(Cl(adj_ohlc_dat)))
  names(ClCL) <- 'y'

  OpCl <- log(Cl(adj_ohlc_dat)) - log(Op(adj_ohlc_dat))
  names(OpCl) <- 'OptoCl_log_returns'

  ClvsMA <- log(Cl(adj_ohlc_dat)) / EMA(log(Cl(adj_ohlc_dat)), n=200)
  names(ClvsMA) <- 'ClvsMA200'

  VolvsMA <- log(Vo(adj_ohlc_dat)) / EMA(log(Vo(adj_ohlc_dat)), n=200)
  VolvsMA[is.na(VolvsMA)] <- 0
  names(VolvsMA) <- 'VolvsMA200'

  out <- merge.xts(ClCL, OpCl, ClvsMA, VolvsMA)

  # Technicals
  DVI = DVI(Cl(adj_ohlc_dat))
  names(DVI) = paste('DVI', names(DVI))
  out <- merge.xts(out, DVI)
  out <- merge.xts(out, CLV(HLC(adj_ohlc_dat)))
  out <- merge.xts(out, ADX(HLC(adj_ohlc_dat)))
  out <- merge.xts(out, KST(Cl(adj_ohlc_dat)))
  out <- merge.xts(out, SAR(cbind(Hi(adj_ohlc_dat),Lo(adj_ohlc_dat))))
  out <- merge.xts(out, TDI(Cl(adj_ohlc_dat)))
  out <- merge.xts(out, aroon(cbind(Hi(adj_ohlc_dat),Lo(adj_ohlc_dat))))
  out <- merge.xts(out, BBands(HLC(adj_ohlc_dat))[,'pctB'])
  out <- merge.xts(out, RSI(Cl(adj_ohlc_dat),2))
  out <- merge.xts(out, CCI(HLC(adj_ohlc_dat)))

  chaikinAD = scale(diff(chaikinAD(HLC(adj_ohlc_dat),Vo(adj_ohlc_dat))), center=FALSE)
  names(chaikinAD) <- 'chaikinAD'
  out <- merge.xts(out, chaikinAD)
  out <- merge.xts(out, CMF(HLC(adj_ohlc_dat),Vo(adj_ohlc_dat)))
  out <- merge.xts(out, MFI(HLC(adj_ohlc_dat),Vo(adj_ohlc_dat)))
  out <- merge.xts(out, VHF(HLC(adj_ohlc_dat)))

  names(out) = gsub('adj_ohlc_dat', '', names(out))
  names(out) = gsub('([[:punct:]]|[[:space:]])+', '.', names(out))
  names(out) = gsub('^\\.', '', names(out))
  names(out) = gsub('\\.$', '', names(out))

  names(out) = make.names(names(out))

  # Remove NAs
  out <- na.omit(out)

  # Return a data.table
  out <- data.table(
    id=n,
    as.data.table(out)
    )
  setnames(out, 'index', 'date')
  out
})

# Make one dataset
tickers_final = rbindlist(tickers_tech)
tickers_final_OTP = copy(tickers_final)
tickers_final_OTP[,y := Next(y, 1)]
tickers_final_OTP <- tickers_final_OTP[!is.na(y),]

# Save
fwrite(tickers_final, '~/datasets/cfds_multiseries/SP500_with_technical_multi.csv')
fwrite(tickers_final_OTP, '~/datasets/cfds_multiseries/SP500_with_technical_otp.csv')
fwrite(tickers_final[,list(id, date, y)], '~/datasets/cfds_multiseries/SP500_multi.csv')
