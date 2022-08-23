###############################################################
#
#   Custom callbacks for DataRobot Keras Models
#
#   Author: Zach Deane-Mayer
#
#   Copyright DataRobot Inc, 2019 onwards
#
#  NOTE: THIS FILE MUST BE LAZILY IMPORTED!
#  NOTE: THIS FILE MUST BE LAZILY IMPORTED!
#  NOTE: THIS FILE MUST BE LAZILY IMPORTED!
###############################################################
from __future__ import absolute_import, division, print_function

import numpy as np
import statsmodels.api as sm
from common import lazy_import
from os import path
import pandas as pd
import time
import keras.backend as K
from keras.callbacks import Callback
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

debugdir = path.expanduser('~')
K = lazy_import('')
DEBUGGING_PLOTS_FLAG = True

class LRFinder(Callback):

    def __init__(self, start_lr=1e-6, end_lr=10.0, number_of_steps=1, beta=0.98):
        super(LRFinder, self).__init__()

        # Definied on init
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.number_of_steps = number_of_steps
        self.beta = beta
        self.lr_multiplier_each_step = (end_lr / start_lr) ** (1.0 / number_of_steps)

        # Defined later
        self.best_loss = 1e9
        self.avg_loss = 0
        self.iteration = 0
        self.losses, self.smoothed_losses, self.learning_rates, self.iterations = [], [], [], []
        self.best_lr = None

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.start_lr)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        self.iteration += 1

        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** self.iteration)

        # Check if the loss is not exploding
        if self.iteration > 1 and smoothed_loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if smoothed_loss < self.best_loss or self.iteration == 1:
            self.best_loss = smoothed_loss

        lr = self.start_lr * (self.lr_multiplier_each_step ** self.iteration)

        self.losses.append(loss)
        self.smoothed_losses.append(smoothed_loss)
        self.learning_rates.append(lr)
        self.iterations.append(self.iteration)

        K.set_value(self.model.optimizer.lr, lr)

    def calculate_smooth_loss(self):
        order = np.argsort(self.learning_rates)[::-1]
        order_loss = np.array(self.losses)[order]
        ordered_lr = np.array(self.learning_rates)[order]
        ordered_log_lr = np.log(ordered_lr)

        ordered_lowess_loss = sm.nonparametric.lowess(
            order_loss, ordered_log_lr, frac=0.15, it=3, return_sorted=True, is_sorted=False
        )

        if DEBUGGING_PLOTS_FLAG:
            t = time.strftime("%Y%m%d-%H%M%S")

            lr_vs_loss = pd.DataFrame({
                'learning_rate': self.learning_rates,
                'loss': self.losses,
            })
            lr_vs_loss.to_csv(path.join(debugdir, 'lr_vs_loss' + t + '.csv'), index=False)

            plt.figure(figsize=(10, 6))
            plt.scatter(ordered_lr, order_loss, facecolors='none', edgecolor='darkblue')
            plt.plot(np.exp(ordered_lowess_loss[:, 0]), ordered_lowess_loss[:, 1], color='black')
            plt.xscale('log')
            plt.savefig(path.join(debugdir, 'learning_rate_vs_loss_with_smoother.png' + t + '.png'))

        return ordered_lowess_loss

    def calculate_max_gradient_learning_rate(self, ordered_lowess_loss):

        lowest_loss_idx = np.argmin(ordered_lowess_loss[:, 1])
        if lowest_loss_idx > 0:
            loss_we_care_about = ordered_lowess_loss[:lowest_loss_idx, :]
        else:
            loss_we_care_about = ordered_lowess_loss

        if loss_we_care_about.shape[0] > 2:
            ordered_smoothed_grad = np.gradient(np.exp(loss_we_care_about[:, 0]), loss_we_care_about[:, 0], edge_order=2)
            best_idx = np.argmin(ordered_smoothed_grad)
            print(best_idx)
        else:
            best_idx = max(0, lowest_loss_idx - 1)

        # TODO: DOC WHY WE EXP THIS
        # MAYBE EXP OUTSIDE THIS FUNCTION
        best_lr = np.exp(loss_we_care_about[best_idx, 0])

        if DEBUGGING_PLOTS_FLAG:
            t = time.strftime("%Y%m%d-%H%M%S")
            plt.figure(figsize=(10, 6))
            plt.scatter(loss_we_care_about[:, 0], ordered_smoothed_grad)
            plt.xscale('log')
            plt.savefig(path.join(debugdir, 'learning_rate_vs_gradient' + t + '.png'))

            plt.figure(figsize=(10, 6))
            plt.plot(np.exp(loss_we_care_about[:, 0]), loss_we_care_about[:, 1], color='black')
            plt.scatter(best_lr, loss_we_care_about[best_idx, 1], facecolors='red', edgecolor='red')
            plt.xscale('log')
            plt.savefig(path.join(debugdir, 'learning_rate_vs_loss_with_best_lr.png' + t + '.png'))

        return best_lr

    def on_train_end(self, logs=None):

        # Smooth the loss
        ordered_lowess_loss = self.calculate_smooth_loss()

        # Calculate the max gradient loss
        best_lr = self.calculate_max_gradient_learning_rate(ordered_lowess_loss)

        # Store the best LR for future reference
        self.best_lr = best_lr

        # Set the model's lr to the best lr
        K.set_value(self.model.optimizer.lr, best_lr)


lrf = LRFinder()
data = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/lr_vs_loss.csv')
lrf.learning_rates = data.learning_rate.values
lrf.losses = data.loss.values
ordered_lowess_loss = lrf.calculate_smooth_loss()
best_lr = lrf.calculate_max_gradient_learning_rate(ordered_lowess_loss)
