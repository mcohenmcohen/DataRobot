####################################################################################
# Setup
####################################################################################

from ModelingMachine.engine.tasks2.keras_deep_learning import KerasRegressor
from ModelingMachine.engine.tasks2.keras_deep_learning import HIDDEN_ACTIVATIONS, OUTPUT_ACTIVATIONS, KERAS_INITS, SIMPLE_ACTIVATIONS

import pickle
import numpy as np
import scipy as sp
from scipy.sparse import load_npz
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import keras.backend as K
from tensorflow import Session, ConfigProto  
K.set_session(Session(
  config=ConfigProto(
    intra_op_parallelism_threads=32, 
    inter_op_parallelism_threads=32)
    ))

# On the beast: source activate py2
# cp /tmp/fit_X.npz /home/datarobot/fit_X.npz
# cp /tmp/fit_y.npy /home/datarobot/fit_y.npy
# cp /home/datarobot/fit_X.npz /tmp/fit_X.npz
# cp /home/datarobot/fit_y.npy /tmp/fit_y.npy
# scp datarobot@10.20.40.30:/home/datarobot/fit_X.npz /Users/zachary/datasets/fit_X.npz
# scp datarobot@10.20.40.30:/home/datarobot/fit_y.npy /Users/zachary/datasets/fit_y.npy
# Files at:
# wget https://s3.amazonaws.com/datarobot_public_datasets/fit_X.npz
# wget https://s3.amazonaws.com/datarobot_public_datasets/fit_y.npy
X = load_npz('/tmp/fit_X.npz')
y = np.load('/tmp/fit_y.npy')

y = np.exp(y)  # OOOOOPS
weights = None
offset = None

np.random.seed(42)
indices = np.random.permutation(X.shape[0])
split = int(X.shape[0]*.80)
train_idx, test_idx = indices[:split], indices[split:]

####################################################################################
# Fit one
####################################################################################

DeepModel = KerasRegressor(
    epochs=4, calibrate=True, log_target=True, loss='mae', hidden_units=[512, 64, 64],
    activation='prelu', learning_rate=0.003, batch_size=32768, max_batch_size=131072,
    double_batch_size=True)
DeepModel.fit(X[train_idx,:], y[train_idx], verbose=1)

pred = DeepModel.predict(X[test_idx,:])
err = np.sqrt(mean_squared_error(np.log1p(pred), np.log1p(y[test_idx])))
print(err)

####################################################################################
# Hyperopt
####################################################################################

X_train = X[train_idx,:]
y_train = y[train_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]

def power_nonzero(x):
  x = int(x)
  if x > 0:
    return(2 ** x)
  return(x)

# https://www.kaggle.com/dreeux/hyperparameter-tuning-using-hyperopt
# https://github.com/hyperopt/hyperopt/wiki/FMin
def objective(space):
  
  h1 = power_nonzero(space['hidden_1'])
  h2 = power_nonzero(space['hidden_2'])
  h3 = power_nonzero(space['hidden_3'])
  
  hidden_units = [h1, h2, h3]
  hidden_units = [h for h in hidden_units if h != 0]
  if len(hidden_units) == 0:
    hidden_units = [2]
  clf = KerasRegressor(
    epochs = int(space['epochs']),
    hidden_units = hidden_units,
    activation = space['activation'],
    final_activation = space['final_activation'],
    learning_rate = space['learning_rate'],
    batch_size = 32768,
    max_batch_size = 131072,
    double_batch_size = space['double_batch_size'],
    calibrate = space['calibrate'],
    log_target = space['log_target'],
    loss = space['model_loss'],
    batch_norm = space['batch_norm'],  # TODO: add output batch norm
    dropout = space['dropout'],
    l1 = space['l1'],
    l2 = space['l2'],
    init_hidden = space['init_hidden'],
    init_output = space['init_output'],
    decay_learning_rate = False,
    pass_through_inputs = space['pass_through_inputs'],
    prediction_clip_min = 0,
    prediction_clip_max = 1000,
  )
  print(clf.get_params())
  print(clf.unit_list)
  print(clf.random_seed_list)
  print(clf.init_list)
  print(clf.dropout_list)
  print(clf.batch_norm_list)
  print(clf.activation_list)
  print(clf.l1_list)
  print(clf.l2_list)
  clf.fit(X_train, y_train, verbose=1)
  pred = clf.predict(X_test)
  pred[np.where(~ np.isfinite(pred))] = 0
  err = np.sqrt(mean_squared_error(np.log1p(pred), np.log1p(y_test)))
  print('RMSLE:' + str(err))
  return {'loss':err, 'status': STATUS_OK}

space = {
  'epochs': hp.quniform('epochs', 1, 10, 1),
  'hidden_1': hp.quniform('hidden_1', 0, 9, 1),
  'hidden_2': hp.quniform('hidden_2', 0, 9, 1),
  'hidden_3': hp.quniform('hidden_3', 0, 9, 1),
  'activation': hp.choice('activation', HIDDEN_ACTIVATIONS),
  'final_activation': hp.choice('final_activation', SIMPLE_ACTIVATIONS),
  'learning_rate': hp.loguniform('learning_rate', np.log(1e-8), np.log(1)),
  'calibrate': hp.choice('calibrate', [True, False]),
  'log_target': hp.choice('log_target', [True, False]),
  'double_batch_size': hp.choice('double_batch_size', [True, False]),
  'model_loss': hp.choice('model_loss', ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error',
                             'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'logcosh', 'poisson', 'gamma']),
  'batch_norm': hp.choice('batch_norm', [True, False]),
  'dropout': hp.uniform('dropout', 0, 1),
  'l1': hp.uniform('l1', 0, 1),
  'l2': hp.uniform('l2', 0, 1),
  'init_hidden': hp.choice('init_hidden', KERAS_INITS),
  'init_output': hp.choice('init_output', KERAS_INITS),
  'pass_through_inputs': hp.choice('pass_through_inputs', [True, False])
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials)
losses = np.array(trials.losses())
print(losses.min())
print(trials.trials[np.argmin(losses)])
print(best)

with open('/tmp/trials3.pickle', 'wb') as handle:
    pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)


# RMSLE:0.5781205
{'dtype': 'float32', 'prediction_clip_min': 0, 'epochs': 5, 'calibrate': True, 'random_seed': 42, 'log_target': True, 'output_batch_norm': False, 'activation': 'elu', 'decay_learning_rate': False, 'optimizer': 'adam', 'learning_rate': 5.0372969294527215e-05, 'double_batch
_size': True, 'batch_size': 32768, 'final_activation': 'linear', 'pass_through_inputs': True, 'dropout': 0.21018238660082428, 'cyclic_learning_rate': False, 'prediction_clip_max': 1000, 'loss': 'mean_absolute_percentage_error', 'hidden_units': [512, 128], 'init_output': '
lecun_normal', 'init_hidden': 'VarianceScaling', 'max_batch_size': 131072, 'l2': 0.9393443944294854, 'l1': 0.4152177003328909, 'batch_norm': True}




{'epochs': 3.0, 'calibrate': 1, 'log_target': 1, 
'learning_rate': 1.9019753568375896e-07, 
'hidden_1': 5.0, 'hidden_3': 6.0, 'hidden_2': 8.0, 
'hidden_5': 0.0, 'hidden_4': 7.0, 'decay_learning_rate': 1, 
'activation': 4, 'double_batch_size': 0, 'batch_size': 18.0, 
'final_activation': 8, 
'pass_through_inputs': 0, 'dropout': 0.16837477676377655, 
'mean_center_std_scale': 0, 'loss': 7, 'init_output': 4, 'init_hidden': 11,
'l2': 0.40668163032272886, 'l1': 0.4094430039441384, 'batch_norm': 0}
