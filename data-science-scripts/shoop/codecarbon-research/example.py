import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# ----------> CODECARBON <-----------
from codecarbon import EmissionsTracker

X, y = datasets.load_diabetes(return_X_y=True)

# Use one feature for training
X = X[:, np.newaxis, 2]

# Split training into train/test splits
X_train = X[:-20]
X_test = X[-20:]

# Split target into train/test splits
y_train = y[:-20]
y_test = y[-20:]

# Linear regression model
model = linear_model.LinearRegression()

# CODECARBON TRACKER (START)
tracker = EmissionsTracker()
tracker.start()

# TRAINING
model.fit(X_train, y_train)

# Make predictions/inference
y_pred = model.predict(X_test)

# CODECARBON TRACKER (STOP)
emissions = tracker.stop()

# Print results
print(f"\nModel used: {model.__class__.__name__}")
print(f"\nY actuals: {y_test}")
print(f"\nY predicted: {y_pred}")
print(f"\nMSE: {mean_squared_error(y_test, y_pred)}")
print(f"\nEmissions: {emissions} kg.")

# Notes from Shoop (06/21/2021)
# After writing up this simple script and running this couple of times, some results for emissions
# that I got included:
# - Emissions: 3.759080166851992e-08 kg 
# - Emissions: 5.159327065636532e-09 kg
