set.seed(42)
x21 <- rgamma(100000,2,10)
x22 <- rgamma(100000,2,1/x21)
x2 <- rpois(100000,(x21*x21+x22+x21*x22)/3)
z21 <- cumsum(x2)
z21 <- z21+1
y2 <- exp(rnorm(80000,9,2))
z2 <- cumsum(y2)
z23 <- z2[z21]
z24 <- c(0,z23)
z25 <- z24[1:100000]
z26 <- z23-z25
z27 <- runif(100000,min=0.8,max=1.2)
target2 <- z26*z27*(5*x21)*sqrt(5*x21)
mydata2 <- data.frame("Target"=target2,"X1"=x21,"X2"=x22)
data.table::fwrite(mydata2, '~/Downloads/tweedie_bug.csv')
