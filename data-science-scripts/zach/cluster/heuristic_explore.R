old_heuristic <- function(n_samples, n_features){
  max_components <- round(0.5*pmin(n_samples, n_features))
  n_components <- round((n_features ** 0.5))
  return(pmin(n_components, max_components))
}

new_heuristic <- function(n_samples, n_features){
  max_components <- pmin(n_samples, n_features)
  n_components <- round(n_features ** 0.5)
  n_components <- max(n_components, 5)
  return(pmin(n_components, max_components))
}

n_samples <- 4000
n_features <- 2

print(old_heuristic(n_samples, n_features))
print(new_heuristic(n_samples, n_features))


 plot(old_heuristic(n_samples, 1:25), type='l')
lines(new_heuristic(n_samples, 1:25), type='l', col='red')

x = 1:25
plot(0.5 * x ~ x, type='l')
lines(sqrt(x) ~ x, type='l')
