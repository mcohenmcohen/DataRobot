# Data from a quadratic polynomial
set.seed(666)
x <- rnorm(100, 5, 2)
y <- (x-5)^2 + rnorm(100)
plot(x, y)

# -- Marginal and non-marginal parametrisations
m.nonmarginal <- lm(y ~ lspline(x, 5))
m.marginal <- lm(y ~ lspline(x, 5, marginal=TRUE))
# Slope of consecutive segments
coef(m.nonmarginal)
# Slope change and consecutive knots
coef(m.marginal)
# Identical predicted values
identical( fitted(m.nonmarginal), fitted(m.marginal))


# -- Different ways to place knots
# Manually: knots at x=4 and x=6
marginal=T
m1 <- lm(y ~ lspline(x, c(4, 6), marginal=marginal))
# 2 knots at terciles of 'x'
m2 <- lm(y ~ qlspline(x, 3, marginal=marginal))
# 3 knots dividing range of 'x' into 4 equal-width intervals
m3 <- lm(y ~ elspline(x, 4, marginal=marginal))
summary(m1)

# Graphically
ox <- seq(min(x), max(x), length=100)
lines(ox, predict(m1, data.frame(x=ox)), col="red")
lines(ox, predict(m2, data.frame(x=ox)), col="blue")
lines(ox, predict(m3, data.frame(x=ox)), col="green")
legend("topright",
       legend=c("m1: lspline", "m2: qlspline", "m3: elspline"),
       col=c("red", "blue", "green"),
       bty="n", lty=1)


function (x, knots = NULL, marginal = FALSE, names = NULL) 
{
  if (!is.null(names)) {
    .NotYetUsed("names")
  }
  n <- length(x)
  nvars <- length(knots) + 1
  namex <- deparse(substitute(x))
  knots <- sort(knots)
  if (marginal) {
    rval <- cbind(x, sapply(knots, function(k) ifelse((x - 
                                                         k) > 0, x - k, 0)))
  }
  else {
    rval <- matrix(0, nrow = n, ncol = nvars)
    rval[, 1] <- pmin(x, knots[1])
    rval[, nvars] <- pmax(x, knots[length(knots)]) - knots[length(knots)]
    if (nvars > 2) {
      for (i in seq(2, nvars - 1)) {
        rval[, i] <- pmax(pmin(x, knots[i]), knots[i - 
                                                     1]) - knots[i - 1]
      }
    }
  }
  colnames(rval) <- seq(1, ncol(rval))
  structure(rval, knots = knots, marginal = marginal, class = c("lspline", 
                                                                "matrix"))
}
<bytecode: 0