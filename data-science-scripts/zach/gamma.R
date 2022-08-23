library(data.table)
x = fread('https://s3-ap-southeast-1.amazonaws.com/datarobotfiles/DR_Demo_Statistical_Case_Estimates.csv')
x[,DateOfAccident := as.Date(DateOfAccident)]
plot(InitialCaseEstimate~DateOfAccident, x)
x[InitialCaseEstimate==235000,]
x[InitialCaseEstimate==235000,as.numeric(DateOfAccident)]

x[,min(DateOfAccident)]
x[,median(DateOfAccident)]
x[,min(DateOfAccident) + 5113]

as.Date(726808, origin = "1970-01-01") - 719891
as.integer(as.Date('2003-06-13'))+716082
as.Date('2003-06-13') - as.Date('1970-01-01') + 719891


4720.57431253 + 
  exp(8.45968574756 + 
        0.9561076327974426*(235000-1000.0)/11366.8769531 +  # InitialCaseEstimate
        -0.04603659798991458*(54-23.0)/106.961524963 +   # ReportingDelay
        0.014753654008637495*(10-11)/3.59781432152 +   # AccidentHour 
        0.3102090334017419*(29-31)/11.9890222549 +   # Age 
        0.44878342531092125*(500-247.039993286)/222.119308472 +   # WeeklyRate 
        0.2253076793026768*(40-38.0)/19.5312786102 +   # HoursWorkedPerWeek 
        0.002023647294769718*(2-0.0)/0.492886930704 +   # DependentChildren 
        -0.002791858289835787*(5-5.0)/0.512922108173 +   # DaysWorkedPerWeek 
        -0.01776312603042734 * (as.integer(as.Date('2003-06-13')) - as.integer(as.Date('1997-07-16'))) / 1492.9552002  + # Date
        0.03996692575570976*(0-0.0)/0.103508807719 # DependentsOther
  )



# http://seananderson.ca/2014/04/08/gamma-glms/
set.seed(999)
N <- 1000
x <- rgamma(N, 10, .1) / 100
y_true <- exp(0.5 + 1.2 * x)
shape <- 10
y <- rgamma(N, rate = shape / y_true, shape = shape)
plot(x, y)
lines(sort(x), y_true[order(x)])

m_glm <- glm(y ~ x, family = Gamma(link = "log"))
m_glm_ci <- confint(m_glm)
coef(m_glm)

plot(predict(m_glm, type='response') ~ log(x))

predict(m_glm, data.frame(x=exp(2)), type='response')
exp(0.5 + 1.2*exp(2))

plot(y~log(x))
