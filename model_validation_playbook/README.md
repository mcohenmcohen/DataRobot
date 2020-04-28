# banking
This repo is comprised of statistical helper functions to support banking model risk management.  Ipython notebooks are provided in many cases to provide the visual display, UI emulation, and examples of the processes.  The code is intended to be taken and implemented/customized for customer environments.
 
## 1. Stress Testing

Continue the range of the input data to include “extreme” values. This will extend partial dependence. 
+ Looking to automate this for the top n features using the API
+ Sent Matt Ben’s notebook and Ofer’s pictures.
+ Ben and Ofer script could be improved by:
  + Parameterizing the inputs 
  + Identifying the original CV records in the original model
  + A more automated way of identifying the blueprint used in the original project.


### Source code
- stress_test.py

  - Retrieve a project's cross-fold validation assignments
  - Retrieve a model's top n features
  - Find the best fitting distribution for a dataset
  - Make a pdf from a distribution and parameters
  - Sample low probability values from a pdf data array/Series 
  - Sample low probability values from a raw data array/Series using KernelDensity

### Notebooks
- fit_best_distribution_and_sample.ipynb
  - Finds the best fitting distribution for a given dataset, and then samples from the distribution to get "extreme" values, statistically expressed as a function of the parameters of the distribution (this last part is TBD).
  
- stress_testing.ipynb
  - Provides a sandbox for the pipeline workflow of a stress test implementation.  E.g.:
    - Get the project, get the cv folds
    - Get the model, get the top n features
    - Create new project(s), get all the xgb blueprints and train those blueprint models using the original cv folds
    - Compare xgb blueprints to those matching in the original project as close as possible, based on the blueprint processes (TBD)
    - then...
    - Find the best fit distribution for the top/applicable features in the data and get extreme values
    - Run stress_test on the xgb models (?) and predict using the derived extreme values
    
## 2. Model validation playbook and automation

We also have the BAML model validation playbook. This includes all of the necessary tests and expectations from model developers for an OLS model. We want to create a script to code these up. We will need to brief prior to starting. 
+ BAML Linear Regression Validation Playbook
+ Many of the tests require the residual
+ Diagnostics Tests (95% confidence level)
  + Linearity 	
  + Heteroscedasticity (statistically)	
  + Independence (serial correlation)	
  + Normality (statistically and visually using histogram or Q-Q Plot)
  + Multicollinearity (correlation, tolerance, VIF)
  + Goodness of Fit
    + R2, Adj_R2, RMSE, F-test
    + T-test, p-value
    + Forward selection method (variable selection)
+ Performance Tests
  + Out-of-Sample Testing	
  + Cross Validation	
  + Sensitivity Testing

### Source code
- ui_utils.py

  - Plots subplots in a variable number of columns and rows
  - Plots of various types for a given model validation specification

### Notebooks
- model_validation_playbook.ipynb
  - Plot visualizations for various linearity tests as prescribed in the Testing Playbook.
  - (TBD) Performs associated statistical tests as prescribed in the Testing Playbook.
