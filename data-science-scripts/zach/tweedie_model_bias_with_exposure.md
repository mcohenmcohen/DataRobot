Tweedie Models, Exposure, and Bias
================

-   [Example data](#example-data)
-   [Model with Exposure and tweedie\_p = 1.5 (the DataRobot
    default)](#model-with-exposure-and-tweedie_p--15-the-datarobot-default)
-   [Solution 1: tweedie\_p = 1.0](#solution-1-tweedie_p--10)
-   [Solution 2: Exposure as weight](#solution-2-exposure-as-weight)
-   [Solution 3: Post proccess the
    model](#solution-3-post-proccess-the-model)
-   [Conclusions](#conclusions)

This document addresses the issue of biased predictions with exposure
models and demonstrates how to replicate DataRobot’s results outside the
platform.

Over prediction (or bias) occurs both with DataRobot **and** outside
DataRobot under the following conditions:

1.  tweedie\_p = 1.5
2.  Model has exposure

Please feel free to used the attached code and data to dig into the
issue. If you setup the problem using the same Target/Exposure/Weight
settings in DataRobot, you will observe the same results both inside and
outside the platform.

I will start by illustrating the “overprediction with exposure” issue,
and then demonstrate 3 different ways to mitigate the overprediction.

## Example data

This
[dataset](https://github.com/datarobot/data-science-scripts/blob/master/zach/tweedie_example.csv)
illustrates this issue:

``` r
library(data.table)
data <- fread('~/workspace/data-science-scripts/zach/tweedie_example.csv')
head(data)
summary(data)
```

| ClaimAmount | Exposure | Idade\_Pessoa\_Segura\_Anuidade | Num\_Participantes | ClaimNb |
|------------:|---------:|--------------------------------:|-------------------:|--------:|
|           0 |    0.504 |                              52 |                682 |       0 |
|           0 |    0.504 |                              58 |               1484 |       0 |
|           0 |    0.290 |                              29 |                 86 |       0 |
|           0 |    0.704 |                               0 |                347 |       0 |
|           0 |    0.164 |                              21 |                343 |       0 |
|           0 |    0.504 |                              11 |                370 |       0 |

|     | ClaimAmount | Exposure     | Idade\_Pessoa\_Segura\_Anuidade | Num\_Participantes | ClaimNb      |
|:----|:------------|:-------------|:--------------------------------|:-------------------|:-------------|
|     | Min. : 0    | Min. : 0.0   | Min. : 0                        | Min. : 1           | Min. :0.00   |
|     | 1st Qu.: 0  | 1st Qu.: 0.3 | 1st Qu.: 24                     | 1st Qu.: 178       | 1st Qu.:0.00 |
|     | Median : 0  | Median : 0.5 | Median : 36                     | Median : 627       | Median :0.00 |
|     | Mean : 36   | Mean : 0.5   | Mean : 35                       | Mean :1259         | Mean :0.02   |
|     | 3rd Qu.: 0  | 3rd Qu.: 0.5 | 3rd Qu.: 47                     | 3rd Qu.:2251       | 3rd Qu.:0.00 |
|     | Max. :85972 | Max. :62.3   | Max. :111                       | Max. :4650         | Max. :9.00   |

The target is `ClaimAmount`, the Exposure is `Exposure`, and the 2
predictors are `Idade_Pessoa_Segura_Anuidade` and `Num_Participantes`.
We will ignore `ClaimNb`.

Lets split this dataset into a training set and a test set:

``` r
set.seed(1234)
data <- data[order(runif(.N)),]
split <- data[,as.integer(.N*.80)]
data_train <- data[1:split,]
data_test  <- data[(split + 1):.N,]
```

## Model with Exposure and tweedie\_p = 1.5 (the DataRobot default)

To start, let’s replicate the “biased predictions with exposure” problem
in pure R code.

The `glm` function in R supports tweedie models. We use the `statmod`
package to supply the `tweedie` family for the `glm` model. In R, we use
`var.power=1.5` in the `tweedie` family function to set tweedie\_p to
1.5. (The “p” in tweedie\_p stands for “power”).

In order to set an exposure (which is multiplicative), we take the
`log(Exposure)` and specify it as an offset.

Since tweedie models use a log link, adding `log(Exposure)` is
equivalent to multiplying by `Exposure`.

``` r
library(statmod)
tweedie_glm_exposure_p1.5 <- glm(
  ClaimAmount ~ Idade_Pessoa_Segura_Anuidade + Num_Participantes,
  family=tweedie(var.power=1.5, link.power=0),
  offset=log(Exposure),
  data=data_train
)
coef(summary(tweedie_glm_exposure_p1.5))
```

|                                 | Estimate | Std. Error | t value | Pr(&gt;\|t\|) |
|:--------------------------------|---------:|-----------:|--------:|--------------:|
| (Intercept)                     |    4.229 |      0.074 |   57.12 |             0 |
| Idade\_Pessoa\_Segura\_Anuidade |    0.027 |      0.002 |   16.65 |             0 |
| Num\_Participantes              |    0.000 |      0.000 |   -5.84 |             0 |

Let’s define a function to evaluate bias. We compare the mean of the
prediction and the mean of the actual. If this number is close to 1, the
model is unbiased. If this number is &lt;1 the model underpredicts, and
if this model is &gt;1, the model overpredicts.

``` r
eval_bias <- function(pred, act){
 return(mean(pred) / mean(act))
}
```

Now lets evaluate the bias of our model on the test set:

``` r
pred_exposure_p1.5 <- predict(tweedie_glm_exposure_p1.5, newdata=data_test, exposure=data_test[,Exposure], type='response')
bias_exposure_p1.5 <- eval_bias(pred_exposure_p1.5, data_test[,ClaimAmount])
print(bias_exposure_p1.5)
```

    ## [1] 2.4

This model is biased! It’s mean prediction is **2.4** times higher than
the actual value of `ClaimAmount`.

If you run this project in DataRobot, you will observe a similar bias,
since we by default use a tweedie\_p of 1.5.

## Solution 1: tweedie\_p = 1.0

One way to solve this problem is to set tweedie\_p to 1.0 for the model
(in DataRobot, this can be done in advanced tuning on the leaderboard.)
In R, we use `var.power=1.0` in the `tweedie` family function to set
tweedie\_p to 1.0. (The “p” in tweedie\_p stands for “power”).

``` r
tweedie_glm_exposure_p1.0 <- glm(
  ClaimAmount ~ Idade_Pessoa_Segura_Anuidade + Num_Participantes,
  family=tweedie(var.power=1.0, link.power=0),
  offset=log(Exposure),
  data=data_train
)
coef(summary(tweedie_glm_exposure_p1.0))
pred_exposure_p1.0 <- predict(tweedie_glm_exposure_p1.0, newdata=data_test, exposure=data_test[,Exposure], type='response')
bias_exposure_p1.0 <- eval_bias(pred_exposure_p1.0, data_test[,ClaimAmount])
print(bias_exposure_p1.0)
```

|                                 | Estimate | Std. Error | t value | Pr(&gt;\|t\|) |
|:--------------------------------|---------:|-----------:|--------:|--------------:|
| (Intercept)                     |    3.193 |      0.123 |   26.06 |             0 |
| Idade\_Pessoa\_Segura\_Anuidade |    0.033 |      0.002 |   13.52 |             0 |
| Num\_Participantes              |    0.000 |      0.000 |   -5.58 |             0 |

    ## [1] 1.02

This model is not biased! It’s mean prediction is **1.02** times higher
than the actual value of `ClaimAmount`.

To replicate this in DataRobot, used advanced tuning to set tweedie\_p =
1.0.

## Solution 2: Exposure as weight

A second solution to this problem is to use the Exposure as a weight
(rather than as an exposure). This is a little more work in DataRobot,
and requires making a new dataset (and therefore a new project):

1.  Divide the target by the exposure  
2.  Set the exposure as a weight

Note that a model with exposure models target/exposure and then
multiplies the prediction by exposure. Dividing the target by exposure
and then multiplying the prediction by exposure replicates this modeling
approach outside of the model.

``` r
tweedie_glm_weight <- glm(
  ClaimAmount / Exposure ~ Idade_Pessoa_Segura_Anuidade + Num_Participantes,
  family=tweedie(var.power=1.5, link.power=0),
  weight=Exposure,
  data=data_train
)
coef(summary(tweedie_glm_weight))
```

|                                 | Estimate | Std. Error | t value | Pr(&gt;\|t\|) |
|:--------------------------------|---------:|-----------:|--------:|--------------:|
| (Intercept)                     |    3.338 |      0.117 |   28.49 |             0 |
| Idade\_Pessoa\_Segura\_Anuidade |    0.029 |      0.003 |   11.27 |             0 |
| Num\_Participantes              |    0.000 |      0.000 |   -4.93 |             0 |

Note that, at prediction time, we need to manually multiply our
prediction by the exposure:

``` r
pred_weight <- predict(tweedie_glm_weight, newdata=data_test, type='response') * data_test[,Exposure]
bias_weight <- eval_bias(pred_weight, data_test[,ClaimAmount])
print(bias_weight)
```

    ## [1] 1.01

This model is not biased! It’s mean prediction is **1.01** times higher
than the actual value of `ClaimAmount`.

Again, to replicate this approach in DataRobot:

1.  Divide the target by exposure  
2.  Use the exposure as a weight  
3.  Multiply DataRobot’s prediction by the exposure

## Solution 3: Post proccess the model

Finally, we can also come up with a scaling factor for our model to
debias it.

``` r
pred_in_sample_exposure_1.5 <- predict(tweedie_glm_exposure_p1.5, data_train, type='response')
ratio <- mean(data_train[,ClaimAmount]) / mean(pred_in_sample_exposure_1.5)
```

At prediction time, we multiple our predictions by this scaling factor:

``` r
pred_exposure_p1.5_adjusted <- pred_exposure_p1.5 * ratio
bias_exposure_p1.5_adjusted <- eval_bias(pred_exposure_p1.5_adjusted, data_test[,ClaimAmount])
print(bias_exposure_p1.5_adjusted)
```

    ## [1] 1.02

This model is not biased! It’s mean prediction is **1.02** times higher
than the actual value of `ClaimAmount`.

You can replicate this approach in DataRobot by predicting on your
training set and calculating the ratio of mean(actual) to
mean(prediction). Then multiply the model’s predictions by this ratio.

In R, we can incorporate this ratio into the model’s intercept:

``` r
tweedie_glm_exposure_p1.5_adjusted <- copy(tweedie_glm_exposure_p1.5)
intercept <- tweedie_glm_exposure_p1.5_adjusted$coefficients[1]
intercept_adjusted <- intercept * ratio
tweedie_glm_exposure_p1.5_adjusted$coefficients[1] <- intercept_adjusted
coef(summary(tweedie_glm_exposure_p1.5_adjusted))
```

|                                 | Estimate | Std. Error | t value | Pr(&gt;\|t\|) |
|:--------------------------------|---------:|-----------:|--------:|--------------:|
| (Intercept)                     |    1.788 |      0.074 |   24.15 |             0 |
| Idade\_Pessoa\_Segura\_Anuidade |    0.027 |      0.002 |   16.65 |             0 |
| Num\_Participantes              |    0.000 |      0.000 |   -5.84 |             0 |

(Note that tweedie models use the log link, which makes their
coefficients + intercept multiplicative)

## Conclusions

So what’s going on here? What’s driving the bias of the first model?
Let’s take a look at the coefficients of all the models to understand
their differences:

|   p | Exposure | Adjusted | Bias | (Intercept) | Idade\_Pessoa\_Segura\_Anuidade | Num\_Participantes |
|----:|:---------|:---------|-----:|------------:|--------------------------------:|-------------------:|
| 1.5 | Exposure | No       | 2.40 |        4.23 |                           0.027 |                  0 |
| 1.0 | Exposure | No       | 1.02 |        3.19 |                           0.033 |                  0 |
| 1.5 | Weight   | No       | 1.01 |        3.34 |                           0.029 |                  0 |
| 1.5 | Exposure | Yes      | 1.02 |        1.79 |                           0.027 |                  0 |

Note that the coefficiemnts from all 4 models are almost identical. The
bias in the predictions from the first model is completely explained by
the intercept of that model. To de-bias the model, we merely need a
different intercept.

The simplest solution to this issue is to change tweedie\_p for your
model from 1.5 to 1.0. In almost all cases, this is a simple fix that
will remove the bias from your models (we are considering making our
default tweedie\_p be 1.0 to address this issue in the future).

A second, slightly more difficult solution is to use exposure as a
weight, instead of as an exposure This is actually a common way to
approach this problem outside of DataRobot, and is a legitimate approach
for building insurance models.

The third solution to this problem is to update your model’s intercept
to remove the bias (or adjust the predictions by a fixed ratio).
