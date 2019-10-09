import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout, LeakyReLU, Input
from tensorflow.keras.models import load_model
from utils import *


class Network:
    def __init__(self, params):
        self.input = params['input_shape']
        self.regularizer = params['regularizer']
        self.dropout = params['dropout']
        self.batch_norm = params['batch_norm']
        self.activation = params['activation']
        self.loss = params['loss']
        self.optimizer = params['optimizer']
        self.metrics = params['metrics']
        self.output = params['output_shape']
        self.mode = params['mode']
        self.save_dir = params['save_dir']
        self.scaler = params['scaler']
        self.train_time = None

        self.model = None

    def trainNetwork(self, params, X_train, Y_train, X_val, Y_val, verbose=False):
        max_steps = params['max_epochs']
        max_idle = params['max_idle']
        batch_size = params['batch_size']

        max_val_metric = starting_metric[self.metrics[0]]
        step_count = 0
        idle = 0
        val_loss = np.inf
        time_start = time.time()
        train_metrics = {x: float() for x in self.metrics}
        val_metrics = {x: float() for x in self.metrics}

        if verbose:
            print("Started training:")
        while keep_training(step_count, idle, val_loss, max_steps, max_idle):
            with tf.device('/gpu:0'):
                step_time = time.time()
                history = self.model.fit(X_train, Y_train, epochs=5, batch_size=batch_size,
                                   validation_data=(X_val, Y_val), verbose=0, shuffle=True)
                # Metrics after current step
                train_loss = np.mean(history.history['loss'])
                val_loss = np.mean(history.history['val_loss'])
                for i in self.metrics:
                    train_metrics[i] = np.mean(history.history[i])
                    val_metrics[i] = np.mean(history.history['val_' + i])

                if compare_metrics(val_metrics[self.metrics[0]], max_val_metric, self.metrics[0]):
                    max_val_metric = val_metrics[self.metrics[0]]
                    idle = 0
                    if self.save_dir:
                        self.save()
                else:
                    idle += 1
                step_count += 1
                if verbose:
                    print('Step {} metrics:'.format(step_count))
                    print('\tLoss --> Train: {}, Val: {}'.format(train_loss, val_loss))
                    for i in self.metrics:
                        print('\t{} --> Train: {}, Val: {}'.format(i, train_metrics[i], val_metrics[i]))
                    print('\tExec. time: {} Idle: {}\n'.format(int(time.time() - step_time), idle))

        self.train_time = time.time() - time_start

    def pred(self, X_test, Y_test, show_mode=None):
        preds = self.model.predict(X_test)
        preds, metrics, score = get_metrics(Y_test, preds, self.mode)

        if show_mode == 'file':
            if self.save_dir:
                with open(self.save_dir + '.txt', 'w') as f:
                    self.model.summary(print_fn=lambda x: f.write(x + '\n'))
                    f.write(metrics)
            else:
                print('Can\'t open file, no save dir specified')

        elif show_mode == 'print':
            self.model.summary()
            print(metrics)

        return preds, score
    
    def save(self):
        if self.save_dir:
            self.model.save(self.save_dir + '.h5')
        else:
            print('Can\'t save, save_dir not defined.')
            
    def load(self):
        if self.save_dir:
            self.model = load_model(self.save_dir + '.h5', custom_objects=custom_objects)
        else:
            print('Can\'t load, save_dir not defined.')


class SequentialNetwork(Network):
    def __init__(self, params):
        super().__init__(params)

        self.model = Sequential()
        self.summary = self.model.summary
        self.fit = self.model.fit
        self.predict = self.model.predict
        self.save = self.model.save

    def add_activation(self):
        if type(self.activation) == str:
            self.model.add(Activation(self.activation))
        else:
            self.model.add(self.activation(alpha=0.01))


class ModularNetwork(Network):
    def __init__(self, params):
        super().__init__(params)

        self.input = Input(shape=self.input)

    def add_activation(self, l):
        if type(self.activation) == str:
            return Activation(self.activation)(l)
        else:
            return self.activation(alpha=0.01)(l)


class FCNN(SequentialNetwork):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']

        self.__compile()

    def __compile(self):  # NN build
        self.model.add(Dense(self.architecture[0], kernel_regularizer=self.regularizer, input_shape=self.input))
        if self.batch_norm:
            self.model.add(BatchNormalization())
        self.add_activation()
        if self.dropout != 0:
            self.model.add(Dropout(self.dropout))
        for i in self.architecture[1:]:
            self.model.add(Dense(i, kernel_regularizer=self.regularizer))
            if self.batch_norm:
                self.model.add(BatchNormalization())
            self.add_activation()
            if self.dropout != 0:
                self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.output))
        self.model.add(Activation(act_dict[self.mode]))

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if self.save_dir:
            self.save()
