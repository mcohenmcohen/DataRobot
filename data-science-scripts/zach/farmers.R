library(data.table)
library(tabplot)
library(ggplot2)
library(scales)
library(glmnet)

data <- fread('~/Downloads/data.csv')
setnames(data, make.names(names(data)))
data_sub <- copy(data)
data_sub <- data_sub[,list(
  DateUnitClosed,
  UnitClosedYear = factor(UnitClosedYear),
  IndemnityPaidAdjusted_5,
  ClaimantAge,
  ClaimantGender=addNA(factor(ClaimantGender)),
  ClaimantRole=addNA(factor(ClaimantRole)),
  InitialClassOfClaim = addNA(factor(InitialClassOfClaim)),
  InjuryCount,
  Fatality = as.integer(Fatality),
  InjuryHealthInsuranceBy = addNA(factor(InjuryHealthInsuranceBy)),
  InjuryLocation_AllOthers,
  InjuryLocation_Chest,
  InjuryLocation_Hand,
  InjuryLocation_Head,
  InjuryLocation_Knee,
  InjuryLocation_LowerArm,
  InjuryLocation_LowerBack,
  InjuryLocation_LowerLeg,
  InjuryLocation_Neck,
  InjuryLocation_Null,
  InjuryLocation_Other,
  InjuryLocation_SoftTissue,
  InjuryLocation_UpperArm,
  InjuryType = factor(make.names(InjuryType)),
  InjurySeverity = factor(make.names(InjurySeverity)),
  InjuryHospital = addNA(factor(make.names(InjuryHospital))),
  LossDescription_Clean = factor(make.names(LossDescription_Clean)),
  PrimaryCause = factor(make.names(PrimaryCause))
)]

data_sub[,list(
  n = .N,
  dollar_sum = sum(IndemnityPaidAdjusted_5),
  med = median(IndemnityPaidAdjusted_5),
  iqr = IQR(IndemnityPaidAdjusted_5),
  mean = mean(IndemnityPaidAdjusted_5),
  sd = sd(IndemnityPaidAdjusted_5)
), by='UnitClosedYear']

data_sub[IndemnityPaidAdjusted_5 > 0,list(
  n = .N,
  dollar_sum = sum(IndemnityPaidAdjusted_5),
  med = median(IndemnityPaidAdjusted_5),
  iqr = IQR(IndemnityPaidAdjusted_5),
  mean = mean(IndemnityPaidAdjusted_5),
  sd = sd(IndemnityPaidAdjusted_5)
), by='UnitClosedYear']

claims_2013 <- data[UnitClosedYear==2013,log10(IndemnityPaidAdjusted_5)]
claims_2014 <- data[UnitClosedYear==2014,log10(IndemnityPaidAdjusted_5)]
claims_2015 <- data[UnitClosedYear==2015,log10(IndemnityPaidAdjusted_5)]
wilcox.test(claims_2013, claims_2015, alternative = c("less"))
wilcox.test(claims_2014, claims_2015, alternative = c("less"))

ggplot(data_sub, aes(x=IndemnityPaidAdjusted_5)) +
  geom_histogram(color='black', fill='white') +
  geom_vline(aes(xintercept=median(IndemnityPaidAdjusted_5, na.rm=T)), color="red", linetype="solid", size=1) +
  geom_vline(aes(xintercept=mean(IndemnityPaidAdjusted_5, na.rm=T)), color="red", linetype="dashed", size=1) +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(labels = comma) + theme_bw() +
  facet_grid(UnitClosedYear~.) +
  ggtitle('Histograms by UnitClosedYear') +
  xlab('IndemnityPaidAdjusted_5: median (solid) and mean (dashed)')

ggplot(data_sub, aes(x=IndemnityPaidAdjusted_5 + 1)) +
  geom_histogram(aes(y=..density..), color='black', fill='white') +
  geom_density(color="black", fill='blue', alpha=.2) +
  geom_vline(aes(xintercept=median(IndemnityPaidAdjusted_5, na.rm=T)), color="red", linetype="solid", size=1) +
  geom_vline(aes(xintercept=mean(IndemnityPaidAdjusted_5, na.rm=T)), color="red", linetype="dashed", size=1) +
  scale_y_continuous(labels = comma) +
  scale_x_log10(labels = comma) + theme_bw() +
  facet_grid(UnitClosedYear~.) +
  ggtitle('Log-Scale Histograms by UnitClosedYear')+
  xlab('IndemnityPaidAdjusted_5: median (solid) and mean (dashed)')

ggplot(data_sub, aes(x=UnitClosedYear, y=IndemnityPaidAdjusted_5+1)) +
  geom_boxplot(outlier.colour = "red", size=1.5) + theme_bw() +
  scale_y_log10(labels = comma) +
  ylab('IndemnityPaidAdjusted_5 (log scale)') +
  ggtitle('Log-Scale Boxplots by UnitClosedYear')

ggplot(data_sub[IndemnityPaidAdjusted_5>0,], aes(x=UnitClosedYear, y=IndemnityPaidAdjusted_5)) +
  geom_boxplot(outlier.colour = "red", size=1.5) + theme_bw() +
  scale_y_log10(labels = comma) +
  ylab('IndemnityPaidAdjusted_5 (log scale)') +
  ggtitle('Log-Scale Boxplots by UnitClosedYear')

tableplot(data_sub, nBins=90)

#GLMNET
yvar <- 'UnitClosedYear'
xvars <- setdiff(names(data_sub), c(yvar, 'DateUnitClosed'))
X <- data_sub[,xvars,with=FALSE]
m <- X[,median(ClaimantAge, na.rm=TRUE)]
X[is.na(ClaimantAge), ClaimantAge := m]
n <- sapply(X, anyNA); n[n]
X <- sparse.model.matrix(~0+., X)
X <- X[,colSums(sign(abs(X)))>10] #Must have 10 obs

set.seed(1)
stopifnot(nrow(X) == nrow(data_sub))
model <- cv.glmnet(
  X,
  data_sub[[yvar]],
  family = 'multinomial',
  alpha = 0.95
)
min(model$cvm)
cf <- coef(model,  model$lambda.1se)
cf <- lapply(cf, function(x){
  x <- x[,1]
  x <- x[abs(x) >= 0.001]
  x <- sort(x)
})

for(n in names(cf)){
  write.csv(
    cf[[n]],
    paste0('~/Desktop/', n, '.csv')
  )
}


cf_2015 <- cf[['2015']][,1]
cf_2015 <- cf_2015[abs(cf_2015) > 0.01]
sort(round(cf_2015, 2))

################################################################################
# OLD
################################################################################


install.packages('tabplot')
library(ggplot2)
library(tabplot)

#http://stackoverflow.com/questions/12865218/getting-rid-of-asis-class-attribute
unAsIs <- function(x) {
  if("AsIs" %in% class(x)) {
    class(x) <- class(x)[-match("AsIs", class(x))]
  }
  x
}
dat_clean <- as.data.frame(lapply(data, unAsIs))
dat_clean$Age_Clean <- as.character(dat_clean$ClaimantAge)
dat_clean$Age_Clean <- sapply(strsplit(dat_clean$Age_Clean, '-'), '[', 1)
dat_clean$Age_Clean <- as.numeric(dat_clean$Age_Clean)


tableplot(
  dat_clean, select_string=c('IndemnityPaidAdjusted_5', 'Age_Clean', 'InjuryType_Fracture', 'InjuryType')
  )
