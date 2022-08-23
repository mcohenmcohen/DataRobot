
binplot <- function(filename){
  library(data.table)
  library(ggplot2)

  dat <- fread(filename)
  cat(readLines(filename, 3), sep='\n')
  setnames(dat, make.names(names(dat)))

  dat <- dat[Transform2=='Binning',list(Feature.Name, Value2, Coefficient)]
  dat[,lower := sapply(strsplit(Value2, ', '), "[", 1)]
  dat[,upper := sapply(strsplit(Value2, ', '), "[", 2)]
  dat[,Value2 := NULL]
  setkeyv(dat, c('Feature.Name', 'lower', 'upper'))

  dat[,lower := gsub("(", "", lower, fixed=TRUE)]
  dat[,upper := gsub(")", "", upper, fixed=TRUE)]
  dat[,upper := gsub("]", "", upper, fixed=TRUE)]
  dat[,lower := as.numeric(lower)]
  dat[,upper := as.numeric(upper)]

  dat[,coef_next := c(Coefficient[2:.N], NA), by='Feature.Name']
  dat[is.na(coef_next), coef_next := Coefficient]

  ggplot(dat, aes(
    x=lower, xend=upper,
    y=Coefficient, yend=Coefficient
    )) +
    geom_segment() +
    geom_segment(aes(
      x=upper, xend=upper,
      y=Coefficient, yend=coef_next)) +
    facet_wrap(~Feature.Name, scales='free') +
    theme_bw() + xlab("Feature Value")
}

binplot("~/Downloads/Elastic-Net Regressor (L2 - Gamma Deviance) with Binned numeric features Feature Coefficients (Fully Transparent) 32.0% Sample Size (1).csv")
