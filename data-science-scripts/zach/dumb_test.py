import pytest
import numpy as np
import pandas as pd
from os import path
import statsmodels.api as sm
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

debugdir = path.expanduser('~')
filepath = 'basic_shape_train.csv'

for (filepath, best_lr) in [
    ('basic_shape_train.csv', 0.01),
	('birds.csv', 0.01),
	('butterflies.csv', 0.01),
	('cats_dogs.csv', 0.01),
	('chest_xray.csv', 0.01),
	('flower.csv', 0.01),
	('food_grocery.csv', 0.01),
	('logo_image.csv', 0.01),
	('monkey_species.csv', 0.01),
	('season_image.csv', 0.01),
]:

    curve = pd.read_csv('tests/tasks2/kerasmodels/test_learning_curves_for_lrf/' + filepath)
    learning_rates = curve['Learning Rate']
    losses = curve['Actual Loss']

    order = np.argsort(learning_rates)[::-1]
    order_loss = np.array(losses)[order]
    ordered_lr = np.array(learning_rates)[order]
    ordered_log_lr = np.log(ordered_lr)

    ordered_lowess_loss = sm.nonparametric.lowess(
        order_loss, ordered_log_lr, frac=0.15, it=3, return_sorted=True, is_sorted=False
    )

    lr_vs_loss = pd.DataFrame({'learning_rate': learning_rates, 'loss': losses})
    idx = np.argmin(ordered_lowess_loss[:, 1])
    used_lr = np.exp(ordered_lowess_loss[idx, 0])/10

    f = interp1d(ordered_lowess_loss[:,0], ordered_lowess_loss[:,1], bounds_error=False)
    used_loss = f(np.log(used_lr))

    plt.figure(figsize=(10, 6))
    plt.scatter(ordered_lr, order_loss, facecolors='none', edgecolor='darkblue')
    plt.plot(np.exp(ordered_lowess_loss[:, 0]), ordered_lowess_loss[:, 1], color='black')
    plt.scatter(np.exp(ordered_lowess_loss[idx, 0]), ordered_lowess_loss[idx, 1], color='red')
    plt.scatter(used_lr, used_loss, color='red')
    plt.xscale('log')
    plt.savefig(path.join(debugdir, 'learning_rate_vs_loss_with_smoother_' + filepath + '.png'))
    plt.close()
