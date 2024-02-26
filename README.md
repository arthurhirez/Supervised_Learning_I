# Regression Models and Supervised Learning I

This repository contains practical assignments from the course "Regression Models and Supervised Learning I." In this course, we explored the fundamentals of linear regression and multiple regression. We covered both the statistical bias to build a theoretical foundation and the computational aspect, developing applications both with and without the assistance of established Python packages and libraries. Bellow there's some examples of few assignments.

# Exercise 1 - Simulation

We adopt the Linear Regression Model:

$$ Y = {\beta}_0 + {\beta}_1 X + \varepsilon, \mbox{ with } \varepsilon \sim \mathcal{N}(0,{\sigma}^2) $$

## Exercise 1.a

For the experiment, we use the following values:

- Parameter ${\beta}_0 = 3$
- Parameter ${\beta}_1 = 5$
- Parameter ${\sigma}^2 = 2^2$
- Number of experiments $N_{exp} = 20$
- Sample size $N = 10$
- Lower limit of the interval $x_{inf} = 0$
- Upper limit of the interval $x_{sup} = 10$

The following table shows the points ${(X_i, Y_i)}$ for the $N$ sample points. It also compares the estimated values of $\hat{\beta}_0$ and $\hat{\beta}_1$, obtained through the Least Squares Method (LSM), and their respective variances with the values defined above in the regression model. Additionally, it calculates the relative difference between the observed values and the calculated values.

| X_obs      | Y_obs      |
|------------|------------|
| 0.000000   | 4.446543   |
| 1.111111   | 8.494903   |
| 2.222222   | 13.694185  |
| 3.333333   | 16.805521  |
| 4.444444   | 25.820357  |

Beta 0:
Observed value:    	2.860	True value:	3.000	Dif:	4.681%
Observed variance:	1.387	Estim. val:	1.176	Dif:	-17.883%

Beta 1:
Observed value:    	4.984	True value:	5.000	Dif:	0.321%
Observed variance:	0.042	Estim. val:	0.033	Dif:	-24.555%

The experiment results can be visualized in the figure below. The left plot shows the sampled points and the linear model fitted to these points through the LSM. The right plot displays all experiments and the line of the model idealized by the defined parameters above, highlighting the variability of the models fitted around the true regression.

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/c03ff50a-8244-46f4-b6b2-6a1efd466ae4)

## Exercise 1.b - Convergence of the beta parameters and their variance with increasing sample size

For the experiment, the following values are adopted:

- Parameter ${\beta}_0 = 3$
- Parameter ${\beta}_1 = 5$
- Parameter ${\sigma}^2 = 2^2$
- Number of experiments $N_{exp} = 500$
- Sample size $N = [10, 25, 50, 100, 150, 250, 500, 1000, 3000, 4000, 5000, 6500, 8000, 10000]$
- Lower limit of the interval $x_{inf} = 0$
- Upper limit of the interval $x_{sup} = 50$

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/823a1fe1-4175-4b61-ab22-ca4808c51161)


## Exercise 1.c - Relationship $S_{xx}$ and Beta Variance
### Experiment Setup
- Parameter ${\beta}_0 = 3$
- Parameter ${\beta}_1 = 5$
- Parameter ${\sigma}^2 = 2^2$
- Number of experiments $N_{exp} = 20$
- Sample size $N = 20$
- Lower limit of the interval $x_{inf} = 0$ (Part I) and $50$ (Part II)
- Upper limit of the interval $x_{sup} = 200$

### Definitions
- ${\sigma}^2$ is the error variance ($\varepsilon \sim \mathcal{N}(0,{\sigma}^2)$);
- $N$ is the sample size;
- $S_{xx} = \sum \limits _{i=1} ^{N} (x_{i} - \bar x)^2 $

### Observations
In the first part of the experiment, the range starts from $X_1 = 0$, varying in increments of 10 units up to $X_n = 200$. In the left plot, it can be observed that the variance $Var[\hat {\beta}_1]$ has low values regardless of the $S_{xx}$ resulting from the variation in the range size. The calculated and observed values are highly similar.

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/95ca40b1-3f42-46a0-83eb-1fe193b86e08)

$Var[\hat {\beta}_0]$ shows some fluctuation around the calculated value. However, in the logarithmic scale plot on the right, a strong correlation can be observed between the calculated and observed variance values of the parameters $\hat {\beta}_i$ as $S_{xx}$ varies. The calculated and observed values are nearly coincident.

Varying the experiment, with the initial range value as $X_1 = 0$, it can be noted that the variance $Var[\hat {\beta}_0]$ starts to have much higher values at the beginning of the experiment, highlighting the difference caused in the intercept when starting a Linear Regression with values far from the origin.

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/ebc6a95d-4d6d-4ab2-b800-ce3ba74dace6)

However, with the increase of the interval, this variation becomes close to what was observed in the first case. This emphasizes that small sampling intervals, whose value of $\hat {\beta}_0$ is high, can lead to large variability for this parameter. Therefore, the observation of this scenario should be treated with caution in Linear Regression modeling.


# Exercise 2 - Residual Analysis

## Exploratory Analysis

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/5ead629b-679d-4df6-986f-4fff14ace439)

## Regression and Residuals

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/95b6ca70-1122-4709-ba01-202e9f2cab75)

It can be observed in the plots below that the quadratic regression model fits much better than the linear model, with the better fit of the quadratic model being evident.

Additionally, it is observed that the errors are much smaller in the case of degree 2 compared to the linear model, also suggesting the better fit of the quadratic model.

|                       | Linear Model          | 2nd Degree Model      |
|-----------------------|-----------------------|------------------------|
| MS - Regression       | 12879.289             | 7106.056               |
| MS - Residuals        | 342.615               | 12.545                |
| R²                    | 0.904                 | 0.997                 |


![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/a775c7a3-a19b-4992-b219-3815b568cc8e)

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/ea6cfe3a-f560-4c7f-9fd4-5b60c29cda7a)

Observing the residuals, it is clear that in the case of linear regression, the residuals show a pattern (parabola) that suggests a higher degree model would be more appropriate. Plotting the residuals for the 2nd-degree regression, an almost random pattern around 0 is observed, highlighting a better fit for this model.

Therefore, analyzing the generated curve, the t-test for the quadratic term, and the residual plot, it is concluded that the 2nd-degree regression model is the most suitable for the case.

## Model Adjustment and t-Statistics

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/cd816bc3-e207-46f5-91a7-df2d5fc49a18)

Apparently, considering the t-test as a criterion, all covariates, given the existence of others in the model, could be discarded. However, this approach allows discarding only one variable, requiring an F-test to decide which variables can be discarded in the model collectively - noting the fact that many covariates have a high correlation, highlighted in the plot below.

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/c336ea82-c6af-4232-9024-bd7eee9569b2)

## c) Residual Analysis

Observing the residuals for each variable, no pattern suggesting non-normality of the residual distribution is detected (although the test with the JB statistic in the ANOVA table above rejects the normality of residuals with 5% significance).

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/6a72e839-d233-43b7-95cd-c8ec6f95de37)

## d) Extra Sum of Squares

Given the high observed correlation, we proceed by conducting an F-test for each of the second-order terms in the model. It is evident that they contribute little compared to the other covariates in the model. Therefore, the decision is made to remove X1^2 and proceed with another F-test.

# Exercise 3 - Polynomial Regression

Apparently, the quadratic model had a better fit to the data, and as a criterion for the lowest regression error, we choose this model (AIC and BIC statistics are very similar for all cases).

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/f26eba21-7cfe-45ee-a033-91085bba21ce)

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/9a6960d6-175b-4934-91da-334cb5985cd7)


|                       | Modelo Linear | Modelo Quadrático | Modelo Polinomial |
|-----------------------|---------------|-------------------|-------------------|
| MS - Regressão        | 0.704         | 0.687             | 0.352             |
| MS - Resíduos         | 0.002         | 0.002             | 0.002             |
| R²                    | 0.349         | 0.340             | 0.349             |


## b)

It is evident, from the correlation matrices below, that simply creating a column equal to the square of another results in high correlation of the data, with a correlation of 0.986 between the variable `Commute` and the variable `Commute^2`.

|           | Commute   | Commute^2 | Mobility  |
|-----------|-----------|-----------|-----------|
| Commute   | 1.000000  | 0.985774  | 0.590634  |
| Commute^2 | 0.985774  | 1.000000  | 0.583288  |
| Mobility  | 0.590634  | 0.583288  | 1.000000  |


By centralizing Z = X - X_mean, the correlation value between Z and Z^2 is only 0.368, a much more favorable situation - since highly correlated columns, almost linearly dependent, result in non-invertible matrices, which can complicate the calculation of the estimated parameter vector.

|            | Commute   | Commute^2 | Z         | Z_quad    | Mobility  |
|------------|-----------|-----------|-----------|-----------|-----------|
| Commute    | 1.000000  | 0.985774  | 1.000000  | 0.367867  | 0.590634  |
| Commute^2  | 0.985774  | 1.000000  | 0.985774  | 0.518922  | 0.583288  |
| Z          | 1.000000  | 0.985774  | 1.000000  | 0.367867  | 0.590634  |
| Z_quad     | 0.367867  | 0.518922  | 0.367867  | 1.000000  | 0.223120  |
| Mobility   | 0.590634  | 0.583288  | 0.590634  | 0.223120  | 1.000000  |

Regarding the model errors, no effect was noticed with the centralization of the variable - the difference will be observed in the next exercise.

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/32a2409b-8b73-48d5-abd0-e79f78f58ae3)

| Estatísticas Resumidas - Modelo Polinomial - Normalizado: |
|---------------------------------------------------------:|
| MS - Regressão:                                       0.352 |
| MS - Resíduos:                                        0.002 |
| R²:                                                    0.349 |


## c)

Observing the estimated coefficient values, it is clear that the data centralization was very efficient, mainly by significantly reducing the standard error of the coefficient estimators.

Polynomial model:
|                   | Coeficiente | Erro Padrão |
|-------------------|-------------|-------------|
| Intercept         | 0.002876    | 0.015583    |
| Commute           | 0.208020    | 0.066933    |
| I(Commute ** 2)   | 0.014239    | 0.067837    |


Normalized polynomial model:
|                   | Coeficiente | Erro Padrão |
|-------------------|-------------|-------------|
| Intercept         | 0.100144    | 0.002063    |
| Z                 | 0.220935    | 0.012098    |
| I(Z ** 2)         | 0.014239    | 0.067837    |


## d)

Observing the errors obtained from the model that separates the regions, it is evident that this change was positive, with a reduction in regression error, while maintaining AIC and BIC very similar to the other cases already presented.

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/4b9f86ac-3b7e-4b94-8bb4-d2f2b796e7d4)

![image](https://github.com/arthurhirez/REGRESSAO/assets/109704516/1e1a72e8-bdf7-43cf-9ac9-dcdfb544ef0a)

Therefore, it is concluded that the change is desirable, and the model with separation by regions is the best option to explain social mobility in this particular case.

