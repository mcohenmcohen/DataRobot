set.seed(42)
x = fread('~/datasets/DR_Demo_Insurance_FreqSevCounts.csv')
x[, ClaimValue := IncurredClaims * rgamma(.N, 2) * 100]
x[which(IncurredClaims==0)[1:100],ClaimValue := 10] # Bad Dayta
fwrite(x, '~/datasets/DR_Demo_Insurance_FreqSevCounts_BadData.csv')
