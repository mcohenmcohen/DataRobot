data {
  int<lower=1> n;
  int<lower=1> k;
  vector[n] x;
  vector[n] y;
  real<lower=0> bw;
  vector[k] omega;
}
transformed data {
  matrix[n,k] cosfeatures;
  matrix[n,k] sinfeatures;
  real scale;
  matrix[n,k] features;

  features = x * omega' * bw;

  scale = sqrt(2.0/n);
  for(i in 1:n)
    for(j in 1:k) {
      cosfeatures[i,j] = cos(features[i,j]);
      sinfeatures[i,j] = sin(features[i,j]);
    }
  cosfeatures = cosfeatures * scale;
  sinfeatures = sinfeatures * scale;
}
parameters {
  vector[k] beta1;
  vector[k] beta2;
  real<lower=0> sigma2;
}
transformed parameters {
  vector[n] fhat;
  fhat = cosfeatures * beta1 + sinfeatures * beta2;
}

model {
  target += normal_lpdf(beta1 | 0, 1);
  target += normal_lpdf(beta2 | 0, 1);
  target += normal_lpdf(sigma2 | 0, 1);
  y ~ normal(fhat, sigma2);
}
