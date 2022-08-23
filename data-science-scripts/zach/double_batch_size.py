class DoubleBatchSize(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_end(self):
        self.batch_size *= 2
        if self.shuffle == True:
            np.random.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y))


training_generator = DoubleBatchSize(x, y, batch_size=32, shuffle=T)
my_model.fix_generator(training_generator)
