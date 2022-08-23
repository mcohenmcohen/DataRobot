class KerasRegressor(BaseEstimator):
    """
    """
    def __init__(self, layers=1, units=16, activation='relu',
                 loss='mean_squared_error', optimizer='adam',
                 epochs=1, final_activation = 'linear',
                 batch_size=2048):
        self.layers = layers
        self.units = units
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.model = None
        self.epochs = epochs
        self.final_activation = final_activation
        self.batch_size = batch_size

        self.graph = None
        self.model_name_on_disk = None

    def __getstate__(self):
        state = self.__dict__.copy()
