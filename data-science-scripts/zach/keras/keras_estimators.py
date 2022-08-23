###############################################################
#
#   Keras-based deep learning models that support sparse input
#
#   Author: Zach Deane-Mayer
#
#   Copyright DataRobot Inc, 2018 onwards
#
###############################################################
from __future__ import division, print_function

import logging
from copy import deepcopy

import numpy as np
from numpy.random import RandomState
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin

import keras
from keras_utils import KERAS_EXPONENTIAL_ACTIVATIONS, \
    KERAS_NEGONE_ONE_ACTIVATIONS, FMIN, FMAX, IMIN, IMAX, KERAS_LOSSES_HIGHER_IS_BETTER, \
    get_activation, get_initializer, get_regularizer, \
    register_custom_keras_functions, log_error_and_raise_to_user, calculate_lr_decay, \
    numpy_inverse_activations, validate_approx_model_size

# Logger
logger = logging.getLogger("datarobot")

# Register my custom activations and losses
register_custom_keras_functions()

#############################################################################
# Docs
#############################################################################

KERAS_DESC = """
Keras is a higher level library for building neural networks using the Tensorflow framework for deep
learning models.  Keras gives more flexability for rapidly incorporating state-of-the-art deep
learning models into DataRobot. Keras also allows for the flexible handling of sparse data, which
can be particularly important for text-heavy data or categorical data with many levels.

Neural networks are a family of models originally inspired by biological neural networks (the
central nervous systems of animals, in particular the brain) and are used to estimate or approximate
functions that can depend on a large number of inputs that are generally unknown. Neural networks
are generally presented as systems of interconnected "neurons" which exchange messages between each
other. The connections have numeric weights that can be tuned based on experience, making neural
nets adaptive to inputs and capable of learning.

A single layer, single-neuron neural network is equivalent to a logistic or linear regression model.
A single-layer, multi-neuron can be thought of as a collection of regression models with an
additional step that learns a linear combination of their output.  All of the models are learned
simultaneously using an optimizer and back-propagation. This form of modeling is very flexible, as
it can learn interesting interactions among the input variables, but it is also much more sensitive
to the input data than regular regression models requiring special techniques such as batch
normalization.

Neural networks in particular are good at finding interactions in data, and this neural network
implementation is particularly good at finding interactions in text data.
"""

KERAS_BASE_PARAMS = """
Parameters
----------
hidden_units : list of ints (default='None')
    Number of units in the hidden layer of the network.  If none, the model is equivalent to a
    simple regression model, fit via SGD, and will not find interactions.  Specify a list of hidden
    units for multiple hidden layers, e.g. list(512, 256, 128) for 3 layers with decreasing
    numbers of units. Use "None" or "list(0)" to fit a model with no hidden layer.
hidden_dropout : list of floats (default='None')
    How much dropout to use for each hidden layer. 0 for no dropout. This regularizes the models and
    makes them less sensitive to input data, but usually causes training to take longer. Be careful
    about setting hidden_batch_norm=True and hidden_dropout>0 at the same time.  If None, will use
    no dropout for all of the hidden layers.
hidden_batch_norm : list of int (default='None')
    Whether or not to batch normalize each hidden layer.  This can speed up model convergence. Be
    careful about setting hidden_batch_norm=True and hidden_dropout>0 at the same time.  1 = use
    batch norm, and 0 = do not use batch norm.  If None, will use no batch norm for all of the
    hidden layers
hidden_l1 : list of float (default=0)
    l1 normalization to use for each hidden layer.  L1 norm will tend to select variables in the
    hidden layer. 0 for no L1 normalization.  If None, will use no L1 for the hidden layers.
    This is a penalty coefficient that is applied to l1(weights) in the loss function.
hidden_l2 : list of float (default=0)
    l2 normalization to use for each hidden layer.  L2 norm will tend to shrink coefficients in the
    hidden layer. 0 for no L2 normalization.  If None, will use no L2 for the hidden layers.
    This is a penalty coefficient that is applied to l2(weights) in the loss function.
hidden_activation : select (default='prelu')
    Activation function to use for the hidden layers only.  "relu" and "prelu" are usually a good
    choice.  Note that while units, hidden_dropout, hidden_batch_norm, hidden_l1, and hidden_l2 are
    lists and can change layer-to-layer, hidden_activation is the same for all hidden layers
hidden_initializer : select (default='he_uniform')
    Initializer for the hidden layer of the model.  Recommended to leave at the default.
output_dropout : float (default=0.0)
    How much dropout to use for the output layer. 0 for no dropout. This regularizes the models and
    makes them less sensitive to input data, but usually causes training to take longer. Be careful
    about setting output_batch_norm=True and output_dropout>0 at the same time.
output_batch_norm : int (default=0)
    Whether or not to batch normalize the output layer.  This can speed up model convergence. Be
    careful about setting output_dropout=True and output_batch_norm>0 at the same time.  1 = use
    batch norm, and 0 = do not use batch norm.
output_l1 : float (default=0.0)
    l1 normalization to use for the output layer.  L1 norm will tend to select variables in the
    hidden layer. 0 for no L1 normalization.
    This is a penalty coefficient that is applied to l1(weights) in the loss function.
output_l2 : float (default=0.0)
    l2 normalization to use for the output layer.  L2 norm will tend to shrink coefficients in the
    hidden layer. 0 for no L2 normalization.
    This is a penalty coefficient that is applied to l2(weights) in the loss function.
output_activation :  select
    Activation for the final output layer of the network.  Recommended to leave at the default.
output_initializer : select (default='he_uniform')
    Initializer for the final output layer of the network.  Recommended to leave at the default.
dropout_type :  select
    Whether to use normal dropout or alphadropout.  Applies to both the hidden layers and the
    output layer.  Use "normal" or "alpha".
loss : select
    Loss function optimized by the model
epochs : int (default=1)
    Number of passes through the data to run.  1 Epoch means the model will consider each point in
    the training data exactly one.
batch_size : int (default=1024)
    The Keras neural networks are trained via SGD, in mini-batches.  This parameter determines how
    many rows to consider for each minibatch.
double_batch_size : bool (default=True)
    If True, the batch size will be doubled every epoch.
max_batch_size : int (default=2 ** 15)
    The maximum batch size the model will consider, to avoid the doubling generating minibatches
    that are too big. Applies only if double_batch_size is set to True
optimizer : select (default='adam')
    Which variant of SGD to use to fit the model. Recommended to use adam.
learning_rate : float (default=0.001)
    Learning rate used for SGD.  Lower learning rates can lead to more accurate models but require
    many more epochs to converge.
random_seed : int (default=42)
    Random seed to use in determining the hidden_dropout.
decay_learning_rate  : bool
    If True, the learning rate will decay every mini-batch, from its initial setting, down to
    a final learning, which by default is 1/10th of its initial setting.  See also
    final_learning_rate_pct
final_learning_rate_pct  : float
    If decay_learning_rate is True, the final learning rate at the end of the epoch.  If
    decay_learning_rate is true, learning rate will decay from the initial learning rate at the
    start of the epoch, down to final_learning_rate_pct*learning_rate at the end of the epoch.
    See also cyclic_learning_rate
cyclic_learning_rate : bool
    If True, reset the learning rate every epoch.  This creates "cycles" of learning in the
    model. Will override decay_learning_rate and set it to True.  This will cause the learning rate
    to decay during a given epoch, and then jump back to the original value at the start of the
    next epoch.  This creates sort of a saw-toothed pattern of learning rates.
pass_through_inputs : bool
    If True, will add a connection of the inputs directly to the output layer.
eps : float (default=0.00000001)
    A small number used for the gamma and tweedie loss functions.
early_stopping : bool
    If True, will check validation loss on a test set with size early_stopping_pct and terminate
    before hitting epochs if that loss increases.
early_stopping_pct : float (default=.1)
    Test set size for early stopping. Must be >0 and <.50.
early_stopping_fine_tune : bool (default=True)
    Boolean.  After early stopping, run one epoch on the early stopping test set, so that the
    test set data gets included in the model.  This usually won't have a big impact on the model,
    but should in most cases give a small accuracy bump.
dtype : select (default='float32')
    INTERNAL PARAMETER, NOT USER-TUNABLE.  Whether to use 'float32' or 'float64' data types
    internally while training the model
"""

KERAS_REG_PARAMS = """

"""

KERAS_REFERENCES = """
References
----------
.. [1] Bishop, Christopher M.
   "Neural networks for pattern recognition."
   Oxford university press, 1995.
   `[link]
   <https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book>`__
.. [2] Dahl, George E., Tara N. Sainath, and Geoffrey E. Hinton. "Improving deep
   neural networks for LVCSR using rectified linear units and dropout." Acoustics,
   Speech and Signal Processing (ICASSP), 2013 IEEE International Conference on. IEEE, 2013.
   `[link]
   <http://www.csri.utoronto.ca/~hinton/absps/georgerectified.pdf>`__
.. [3] Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of
   feature detectors." arXiv preprint arXiv:1207.0580 (2012).
   `[link]
   <http://arxiv.org/pdf/1207.0580.pdf?utm_content=buffer3e047>`__
.. [4] Sergey Ioffe, Christian Szegedy.
   "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
   Covariate Shift." JMLR: Workshop and Conference Proceedings. No. 32. 2015.
   `[link]
   <http://jmlr.org/proceedings/papers/v37/ioffe15.pdf>`__
.. [5] Diederik Kingma, Jimmy Ba.
   "Adam: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980 (2015)
   `[link]
   <http://arxiv.org/pdf/1412.6980v8.pdf>`__
.. [6] Martin Abadi, et al. "TensorFlow: Large-Scale Machine Learning on Heterogeneous
   Distributed Systems."
   `[link]
   <http://download.tensorflow.org/paper/whitepaper2015.pdf>`__
.. [7] Quoc V. Le, et al. "On optimization methods for deep learning."
    Paper presented at the meeting of the ICML, 2011.
    `[link]
    <http://ai.stanford.edu/~quocle/LeNgiCoaLahProNg11.pdf>`__
"""

KERAS_SEE_ALSO = """
See Also
--------
External:
    `Artificial neural network wikipedia
    <https://en.wikipedia.org/wiki/Artificial_neural_network>`_

External:
    `Keras official webpage
    <https://keras.io>`_

External:
    `Keras on github
    <https://github.com/fchollet/keras>`_

External:
    `TensorFlow official webpage
    <https://www.tensorflow.org/>`_

External:
    `TensorFlow on github
    <https://github.com/tensorflow/tensorflow>`_
"""


#############################################################################
# Base sklearn estimators
#############################################################################

def build_dense_layer(input_tensor, hidden_init, l1, l2, act_string, units, bias_init, use_bias,
                      batch_norm, dropout, dropout_type, random_state, name='layer'):
    """
    Function to build a simple, Dense, deep learning layer

    Uses the following components:
    - Initializer (needs a random state for reproducibility)
    - Dense layer (with optional L1 and/or L2 regularization on the weights)
    - Activation
    - Batch norm
    - Dropout (needs a random state for reproducibility)
    """

    # Choose initializer
    hidden_init = get_initializer(hidden_init, random_state.randint(low=IMIN, high=IMAX))

    # Choose regularizer
    reg = get_regularizer(l1, l2)

    # Add the dense layer
    act_dict = get_activation(act_string)
    output_tensor = keras.layers.Dense(
        units=units, activation=act_dict['act_string'], kernel_regularizer=reg,
        kernel_initializer=hidden_init, bias_initializer=bias_init, use_bias=use_bias,
        name=name)(input_tensor)

    # Add advanced activations
    if act_dict['advanced_act']:
        output_tensor = act_dict['advanced_act']()(output_tensor)

    # Add Batch Norm
    if batch_norm == 1:
        output_tensor = keras.layers.BatchNormalization()(output_tensor)

    # Add Dropout
    if dropout > 0:
        s = random_state.randint(low=IMIN, high=IMAX)
        if dropout_type == 'normal':
            output_tensor = keras.layers.Dropout(rate=dropout, seed=s)(output_tensor)
        elif dropout_type == 'alpha':
            output_tensor = keras.layers.AlphaDropout(rate=dropout, seed=s)(output_tensor)
        else:
            log_error_and_raise_to_user('Invalid Dropout Type', {'dropout_type': dropout_type})

    return output_tensor


def none_or_zero_len(x):
    """
    We use None as the defaults for out list parameters.
    We can't use [] as a default, because lists as defaults are bad
    So we use None as the default, and make it mean the same thing as []

    NOTE:
    base modeler passes 'list()' instead of None, which is insane
    list() parses through blueprint interpriter to [].
    """
    return (x is None) or len(x) == 0


def pad_list_with_zeros_if_empty(param, length, outvalue=0):
    """
    If a list param is None or [], replace it with a list of zeroes of a certain length.

    Otherwise hyperparameter is left unchanged.

    This makes it easier to specify list parameters.  For example for a model with no
    dropout, you can specify dropout=None or dropout=[], rather than dropout=[0, 0, 0]
    """
    if none_or_zero_len(param):
        return [outvalue] * length
    return param


class BaseKeras(BaseEstimator):

    def __init__(
        self,
        hidden_units=None,
        hidden_dropout=None,
        hidden_batch_norm=None,
        hidden_l1=None,
        hidden_l2=None,
        hidden_activation='prelu',
        hidden_initializer='he_uniform',
        output_dropout=0.0,
        output_batch_norm=0,
        output_l1=0.0,
        output_l2=0.0,
        output_activation='linear',
        output_initializer='he_uniform',
        dropout_type='normal',
        loss='mean_squared_error',
        epochs=1,
        batch_size=1024,
        double_batch_size=False,
        max_batch_size=2 ** 15,
        optimizer='adam',
        learning_rate=0.001,
        random_seed=42,
        decay_learning_rate=False,
        final_learning_rate_pct=.10,
        cyclic_learning_rate=False,
        pass_through_inputs=False,
        early_stopping=False,
        early_stopping_pct=0.1,
        early_stopping_fine_tune=True,
        dtype='float32'
    ):

        # If hidden_units is None or [], we have no hidden layers
        if none_or_zero_len(hidden_units):
            hidden_units = []

        # For hidden_dropout, batch norm, l1, and l2 use 0 for all hidden layers if None or []
        n_hidden = len(hidden_units)
        hidden_dropout = pad_list_with_zeros_if_empty(hidden_dropout, n_hidden)
        hidden_batch_norm = pad_list_with_zeros_if_empty(hidden_batch_norm, n_hidden)
        hidden_l1 = pad_list_with_zeros_if_empty(hidden_l1, n_hidden)
        hidden_l2 = pad_list_with_zeros_if_empty(hidden_l2, n_hidden)

        # Parameters specified explicitly at init
        self.hidden_units = hidden_units
        self.hidden_dropout = hidden_dropout
        self.hidden_batch_norm = hidden_batch_norm
        self.hidden_l1 = hidden_l1
        self.hidden_l2 = hidden_l2
        self.hidden_activation = hidden_activation
        self.hidden_initializer = hidden_initializer
        self.output_dropout = output_dropout
        self.output_batch_norm = output_batch_norm
        self.output_l1 = output_l1
        self.output_l2 = output_l2
        self.output_activation = output_activation
        self.output_initializer = output_initializer
        self.dropout_type = dropout_type
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.double_batch_size = double_batch_size
        self.max_batch_size = max_batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.decay_learning_rate = decay_learning_rate
        self.final_learning_rate_pct = final_learning_rate_pct
        self.cyclic_learning_rate = cyclic_learning_rate
        self.pass_through_inputs = pass_through_inputs
        self.early_stopping = early_stopping
        self.early_stopping_pct = early_stopping_pct
        self.early_stopping_fine_tune = early_stopping_fine_tune
        self.dtype = dtype

        # Magic parameters, that are overridden at fit time by the MultiClass mixin
        self.output_dim = 1

        # Parameters specified implicitly, depending on the explicit init paramters
        self.random_state = RandomState(seed=self.random_seed)

        # Decide if we will need to clip the predictions at prediction time
        # Exponential models are prone to infinities, so clip them at the top
        self.prediction_clip_min = None
        self.prediction_clip_max = None
        if self.output_activation in KERAS_EXPONENTIAL_ACTIVATIONS:
            self.prediction_clip_max = FMAX

        # Parameters determined at fit time
        self.input_dim = None
        self.sparse = False
        self.model = None
        self.target_center = 0  # Will not change the target.  Can be overwritten at fit time
        self.n_batches = None
        self.has_offset = None

    def validate(self):

        # Check number of hidden layers
        if len(self.hidden_units) > 25:
            log_error_and_raise_to_user(
                'Keras models must have 25 or fewer layers.',
                extra={'layers': len(self.hidden_units)})

        # Check output units
        if self.output_dim <= 0:
            log_error_and_raise_to_user(
                'Final output units for keras model is the output layer, and must be >0',
                extra={'output_units': self.output_dim})

        # Check cyclic learning rate
        if self.cyclic_learning_rate:
            if not self.decay_learning_rate:
                logger.warn('Setting decay_learning_rate=True since cyclic_learning_rate=True.')
            self.decay_learning_rate = True

        # Check pass_through_inputs
        if self.pass_through_inputs and len(self.hidden_units) == 1:
            logger.warn('Ignoring passthrough since there is no hidden layer')
            self.pass_through_inputs = False

        # Check hidden units
        if np.any(np.array(self.hidden_units) <= 0):
            log_error_and_raise_to_user(
                'Hidden_units must all be > 0',
                {'hidden_units': self.hidden_units}
            )

        # Check dropout
        dropout_array = np.array(self.hidden_dropout)
        if np.any(np.logical_or(dropout_array < 0, dropout_array > 1)):
            log_error_and_raise_to_user(
                'hidden_dropout must be all be >= 0 and <= 1.',
                {'hidden_dropout': self.hidden_dropout}
            )
        if self.dropout_type not in ['normal', 'alpha']:
            log_error_and_raise_to_user('Invalid Dropout Type', {'dropout_type': self.dropout_type})

        # Check hidden_l1
        if np.any(np.array(self.hidden_l1) < 0):
            log_error_and_raise_to_user(
                'hidden_l1 must be all be >= 0',
                {'hidden_l1': self.hidden_l1}
            )

        # Check hidden_l2
        if np.any(np.array(self.hidden_l2) < 0):
            log_error_and_raise_to_user(
                'hidden_l2 must be all be >= 0',
                {'hidden_l2': self.hidden_l2}
            )

        # Check list params are all lists and all have the same length
        list_params = {'hidden_units': self.hidden_units,
                       'hidden_dropout': self.hidden_dropout,
                       'hidden_batch_norm': self.hidden_batch_norm,
                       'hidden_l1': self.hidden_l1,
                       'hidden_l2': self.hidden_l2,
                       }
        base_msg = 'For a Keras model'
        for key, value in list_params.items():
            if not isinstance(value, list):
                err = '{}, {} must be a list.'.format(base_msg, key)
                log_error_and_raise_to_user(err, extra=list_params)
            if len(value) != len(self.hidden_units):
                err = base_msg + key + ' must have the same length as hidden_units.'
                log_error_and_raise_to_user(err, extra=list_params)

        # Check early stopping
        if self.early_stopping_pct < 0 or self.early_stopping_pct > 0.5:
            log_error_and_raise_to_user(
                'Early stopping pct must be >0.00 and <0.50',
                extra={'pct': self.early_stopping_pct}
            )
        if self.early_stopping and self.epochs < 3:
            self.early_stopping = False
            logger.warn("Can't do early stopping with fewer than 3 epochs.  Setting "
                        "early_stopping=False.")

        # Check that input dimension is known
        if not self.input_dim:
            raise ValueError('Input dimension not known.')

        # Check that sparsity is known
        if self.sparse is None:
            raise ValueError('Data Sparsity not known.')

        # Check dtype is known
        if not self.dtype:
            raise ValueError('Data dtype not known.')

        # Check hidden units are known
        if np.any(np.array(self.hidden_units) <= 0):
            err = 'Hidden units must all be > 0'
            logger.error(err, extra={'units': self.hidden_units})
            raise ValueError(err + ' Got {}'.format(self.hidden_units))

        # Check offset is known
        if self.has_offset is None:
            raise ValueError('Offset presence / absence not known.')

    def add_inputs(self):
        """ Define the input tensor for a network"""
        input_tensor = keras.layers.Input(shape=(self.input_dim,), sparse=self.sparse,
                                          dtype=self.dtype,
                                          name='input-layer')

        return input_tensor

    def add_dense_layers(self, input_tensor):
        """ Add zero or more Dense layers to the network.  If self.hidden_units=None, we will
        essentially skip this step"""
        output_tensor = input_tensor
        for layer_index, hu in enumerate(self.hidden_units):
            output_tensor = build_dense_layer(
                input_tensor=output_tensor,
                hidden_init=self.hidden_initializer,
                l1=self.hidden_l1[layer_index],
                l2=self.hidden_l2[layer_index],
                act_string=self.hidden_activation,
                units=hu,
                bias_init='zeros',
                use_bias=True,
                batch_norm=self.hidden_batch_norm[layer_index],
                dropout=self.hidden_dropout[layer_index],
                dropout_type=self.dropout_type,
                random_state=self.random_state,
                name='hidden-layer' + str(layer_index)
            )
        return output_tensor

    def add_output_layer(self, input_tensor):
        """ Add output layer to the network."""
        # Add the output layer
        # Use linear activation, so we can add the offset (if specified) prior to final activation
        output_tensor = build_dense_layer(
            input_tensor=input_tensor,
            hidden_init=self.output_initializer,
            l1=self.output_l1,
            l2=self.output_l2,
            act_string='linear',
            units=self.output_dim,
            bias_init=keras.initializers.Constant(self.target_center),
            use_bias=True,
            batch_norm=self.output_batch_norm,
            dropout=self.output_dropout,
            dropout_type=self.dropout_type,
            random_state=self.random_state,
            name='output-layer'
        )

        # Add pass through of input to output
        if self.pass_through_inputs:
            passthrough_tensor = build_dense_layer(
                input_tensor=input_tensor,
                hidden_init=self.output_initializer,
                l1=self.output_l1,
                l2=self.output_l2,
                act_string='linear',
                units=self.output_dim,
                bias_init='zeros',
                use_bias=False,
                batch_norm=self.output_batch_norm,
                dropout=self.output_dropout,
                dropout_type=self.dropout_type,
                random_state=self.random_state,
                name='pass-through-layer')
            output_tensor = keras.layers.Add(
                name='pt-output-add-layer')([output_tensor, passthrough_tensor])

        # Add offset
        if self.has_offset:
            offset_input = keras.layers.Input(shape=(self.output_dim,), dtype=self.dtype,
                                              name='offset-input-layer')
            output_tensor = keras.layers.Add(
                name='offset-output-add-layer')([output_tensor, offset_input])

        # Add final activation
        if self.output_activation != 'linear':
            output_tensor = keras.layers.Activation(self.output_activation)(output_tensor)

        return output_tensor

    # noinspection PyInterpreter
    def build_model(self):
        """ Combine all the layers to build an MLP """

        input_tensor = self.add_inputs()

        output_tensor = self.add_dense_layers(input_tensor)

        output_tensor = self.add_output_layer(output_tensor)

        self.model = keras.models.Model(input_tensor, output_tensor)

    def compile_model(self):
        config = {'lr': self.learning_rate}

        if self.decay_learning_rate:
            if self.n_batches > 0:
                decay = calculate_lr_decay(self.learning_rate,
                                           self.final_learning_rate_pct * self.learning_rate,
                                           self.n_batches)
                if self.optimizer == 'nadam':
                    config['schedule_decay'] = decay
                else:
                    config['decay'] = decay
            else:
                logger.warn('self.n_batches is unknown.  Not setting decay')
        opt = keras.optimizers.get(
            {'class_name': self.optimizer, 'config': config})
        self.model.compile(optimizer=opt, loss=self.loss)

    def freeze_model(self):
        for layer in self.model.layers:
            layer.trainable = False

    def thaw_model(self):
        for layer in self.model.layers:
            layer.trainable = True

    def copy_model(self):
        model_copy = keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())
        return model_copy

    # Overridable function to determine the center of the target
    def compute_center(self, y, sample_weight=None):
        center = np.average(y, axis=0, weights=sample_weight)
        inv_center = numpy_inverse_activations[self.output_activation](center)
        inv_center = np.clip(inv_center, FMIN, FMAX)
        if not np.isfinite(inv_center):
            inv_center = 0
        self.target_center = inv_center

    def preprocess_inputs(self, X, y=None, **kwargs):
        if self.sparse:
            X = csr_matrix(X)
        return X, y

    def fit(self, X, y, sample_weight=None, offset=None, verbose=False):

        # Set Input dimension
        self.input_dim = X.shape[1]

        # Set offset
        self.has_offset = offset is not None

        # If input is a csc or coo, convert to csr
        if isinstance(X, csc_matrix) or isinstance(X, coo_matrix):
            self.sparse = True
        # If input is numpy array, only convert to csr if it is mainly zero
        elif isinstance(X, np.ndarray):
            if np.count_nonzero(X) / X.size < .10:
                self.sparse = True

        # Preprocess data (e.g. reshape, convert to sparse if needed)
        X, y = self.preprocess_inputs(X, y, train_mode=True)

        # Determine batch size
        training_rows = X.shape[0]
        self.n_batches = int(np.ceil(training_rows / self.batch_size))

        # Make sure our inputs have the right dtype
        if offset is not None:
            assert isinstance(offset, np.ndarray)
            assert np.all(np.isfinite(offset))
            assert offset.shape[1] == self.output_dim
            X = [X, offset]
        if sample_weight is not None:
            assert isinstance(sample_weight, np.ndarray)
            assert np.all(np.isfinite(sample_weight))

        # Calculate the target scale we're gonna use
        self.compute_center(y, sample_weight)

        # Now that we know all the modeling parameters, validate the model
        self.validate()

        # Construct the keras model
        self.build_model()

        # Validate we'll be able to pickle the model
        validate_approx_model_size(self.model, self.batch_size, self.dtype)

        # Calculate early stopping
        train_indexes = np.arange(training_rows)
        test_indexes = None
        best_valid_error = None
        if self.early_stopping:
            self.random_state.shuffle(train_indexes)
            N = train_indexes.shape[0]
            breakpoint = int(np.round(N * self.early_stopping_pct))
            test_indexes = train_indexes[0:breakpoint]
            train_indexes = train_indexes[breakpoint:N]

        # Subset sample weights
        if sample_weight is not None:
            sample_weight_train = sample_weight[test_indexes].ravel()
            sample_weight_test = sample_weight[test_indexes].ravel()
        else:
            sample_weight_train = None
            sample_weight_test = None

        # Fit the model
        batch_size = self.batch_size
        for i in range(self.epochs):

            # Always compile the model before the first epoch
            # If cyclic_learning_rate, recompile each epoch to reset the LR
            if i == 0 or self.cyclic_learning_rate:
                self.compile_model()

            # Shuffle the data
            self.random_state.shuffle(train_indexes)

            # Fit the model
            self.model.fit(
                X[train_indexes, :], y[train_indexes], sample_weight=sample_weight_train,
                batch_size=batch_size, epochs=1, shuffle=False, verbose=verbose)

            # Increase the batch size
            if self.double_batch_size:
                if batch_size * 2 < self.max_batch_size:
                    batch_size *= 2
                else:
                    batch_size = self.max_batch_size

            # Check early stopping
            # TODO: add patience
            # https://datarobot.atlassian.net/browse/MODEL-191
            if self.early_stopping:
                current_val_error = self.model.evaluate(
                    X[test_indexes, :], y[test_indexes], sample_weight=sample_weight_test,
                    batch_size=batch_size, verbose=verbose)

                # Make loss always lower == better
                if self.loss in KERAS_LOSSES_HIGHER_IS_BETTER:
                    current_val_error = -1 * current_val_error

                # If this is the first loop, set best error to current error
                if best_valid_error is None:
                    best_valid_error = deepcopy(current_val_error)

                # If current error is better, set best error to current error
                elif current_val_error <= best_valid_error:
                    best_valid_error = deepcopy(current_val_error)

                # Current error is worse than best error.  Break the training loop
                else:
                    logger.info("Keras model stopping early.", extra={
                        'Initial epochs': self.epochs,
                        'Stopped at': i,
                        'Current loss': current_val_error,
                        'Best Loss': current_val_error,
                    })
                    break

        # Update the model on the test set
        if self.early_stopping and self.early_stopping_fine_tune:
            self.model.fit(
                X[test_indexes, :], y[test_indexes], sample_weight=sample_weight_test,
                batch_size=batch_size, epochs=1, shuffle=False, verbose=verbose)

        # Return
        return self

    def predict(self, X, offset=None, *args, **kwargs):
        X, _ = self.preprocess_inputs(X, train_mode=False)
        if self.has_offset:
            if not offset:
                offset = np.ones(X.shape[0])
            X = [X, offset]
        pred = self.model.predict(X, batch_size=self.batch_size)
        if self.prediction_clip_min or self.prediction_clip_max:
            pred = np.clip(pred, self.prediction_clip_min, self.prediction_clip_max, out=pred)
        return pred


#############################################################################
# sklearn estimators
#############################################################################


class KerasRegressor(BaseKeras, RegressorMixin):

    def __init__(
        self,
        hidden_units=None,
        hidden_dropout=None,
        hidden_batch_norm=None,
        hidden_l1=None,
        hidden_l2=None,
        hidden_activation='prelu',
        hidden_initializer='he_uniform',
        output_dropout=0.0,
        output_batch_norm=0,
        output_l1=0.0,
        output_l2=0.0,
        output_activation='linear',
        output_initializer='he_uniform',
        dropout_type='normal',
        loss='mean_squared_error',
        epochs=1,
        batch_size=1024,
        double_batch_size=False,
        max_batch_size=2 ** 15,
        optimizer='adam',
        learning_rate=0.001,
        random_seed=42,
        decay_learning_rate=False,
        final_learning_rate_pct=.10,
        cyclic_learning_rate=False,
        pass_through_inputs=False,
        early_stopping=False,
        early_stopping_pct=0.1,
        early_stopping_fine_tune=True,
        dtype='float32'
    ):
        super(KerasRegressor, self).__init__(
            hidden_units=hidden_units,
            hidden_dropout=hidden_dropout,
            hidden_batch_norm=hidden_batch_norm,
            hidden_l1=hidden_l1,
            hidden_l2=hidden_l2,
            hidden_activation=hidden_activation,
            hidden_initializer=hidden_initializer,
            output_dropout=output_dropout,
            output_batch_norm=output_batch_norm,
            output_l1=output_l1,
            output_l2=output_l2,
            output_activation=output_activation,
            output_initializer=output_initializer,
            dropout_type=dropout_type,
            loss=loss,
            epochs=epochs,
            batch_size=batch_size,
            double_batch_size=double_batch_size,
            max_batch_size=max_batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            random_seed=random_seed,
            decay_learning_rate=decay_learning_rate,
            final_learning_rate_pct=final_learning_rate_pct,
            cyclic_learning_rate=cyclic_learning_rate,
            pass_through_inputs=pass_through_inputs,
            early_stopping=early_stopping,
            early_stopping_pct=early_stopping_pct,
            early_stopping_fine_tune=early_stopping_fine_tune,
            dtype=dtype
        )


class KerasClassifier(BaseKeras, ClassifierMixin):

    def __init__(
        self,
        hidden_units=None,
        hidden_dropout=None,
        hidden_batch_norm=None,
        hidden_l1=None,
        hidden_l2=None,
        hidden_activation='prelu',
        hidden_initializer='he_uniform',
        output_dropout=0.0,
        output_batch_norm=1,  # Changed from Base
        output_l1=0.0,
        output_l2=0.0,
        output_activation='sigmoid',  # Changed from Base
        output_initializer='he_normal',  # Changed from Base
        dropout_type='normal',
        loss='binary_crossentropy',  # Changed from Base
        epochs=1,
        batch_size=1024,
        double_batch_size=False,
        max_batch_size=2 ** 15,
        optimizer='adam',
        learning_rate=0.001,
        random_seed=42,
        decay_learning_rate=False,
        final_learning_rate_pct=.10,
        cyclic_learning_rate=False,
        pass_through_inputs=False,
        early_stopping=False,
        early_stopping_pct=0.1,
        early_stopping_fine_tune=True,
        dtype='float32'
    ):
        super(KerasClassifier, self).__init__(
            hidden_units=hidden_units,
            hidden_dropout=hidden_dropout,
            hidden_batch_norm=hidden_batch_norm,
            hidden_l1=hidden_l1,
            hidden_l2=hidden_l2,
            hidden_activation=hidden_activation,
            hidden_initializer=hidden_initializer,
            output_dropout=output_dropout,
            output_batch_norm=output_batch_norm,
            output_l1=output_l1,
            output_l2=output_l2,
            output_activation=output_activation,
            output_initializer=output_initializer,
            dropout_type=dropout_type,
            loss=loss,
            epochs=epochs,
            batch_size=batch_size,
            double_batch_size=double_batch_size,
            max_batch_size=max_batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            random_seed=random_seed,
            decay_learning_rate=decay_learning_rate,
            final_learning_rate_pct=final_learning_rate_pct,
            cyclic_learning_rate=cyclic_learning_rate,
            pass_through_inputs=pass_through_inputs,
            early_stopping=early_stopping,
            early_stopping_pct=early_stopping_pct,
            early_stopping_fine_tune=early_stopping_fine_tune,
            dtype=dtype
        )

    def fit(self, X, y, sample_weight=None, offset=None, **kwargs):
        # Map 0/1 to -1/1
        # 0/1 - 0.5 = -.5/.5
        # -.5/+.5 / 0.50 = -1/1
        if self.output_activation in KERAS_NEGONE_ONE_ACTIVATIONS:
            y = (y - 0.50) / 0.50
        super(KerasClassifier, self).fit(X, y, sample_weight=sample_weight, offset=offset, **kwargs)

    def predict_proba(self, X, offset=None, *args, **kwargs):
        pred = super(KerasClassifier, self).predict(X, offset=offset, *args, **kwargs)
        # Map -1/1 back to 0/1
        # -1/1 * 0.50 = -.5/.5
        # -.5/.5 + 0.50 = 0/1
        if self.output_activation in KERAS_NEGONE_ONE_ACTIVATIONS:
            pred = pred * 0.5 + 0.50
        return pred

    def predict(self, X, offset=None, *args, **kwargs):
        # TODO: I really feel like this should predict 0/1
        # but doing so appears to break our modelers
        # https://datarobot.atlassian.net/browse/MODEL-192
        return self.predict_proba(X, offset=offset, *args, **kwargs)
