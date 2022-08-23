# Make some bimodal data
set.seed(42)
N <- 100
x <- sort(rbeta(N, .2, .2))
hist(x)

# Find the percentiles of that data
number_of_buckets = 10
probs <- seq(0, 1 ,by=1/number_of_buckets)
quants <- quantile(x, probs, rule=2)

# Plot the relationship between the probabilities and the quantiles
# The quantile values are the input data, and the probabilities are the outputs
plot(probs ~ quants)
lines(probs ~ quants)

# Map each point to the closest percentile (and interpolate between them)
# This turns our bimodal distribution into a uniform distribution
smooshed_data <- approx(quants, probs, xout=x)$y
hist(smooshed_data)

# Function to do the whole process
simplified_ridit <- function(x){
  probs <- seq(0, 1 ,by=.01)
  quants <- quantile(x, probs, rule=2)
  approx(quants, probs, xout=x)$y
}
plot(simplified_ridit(x) ~ x, type='l')
