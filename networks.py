import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from utils import *


class Network:
    def __init__(self, params):
        self.input_shape = params['input_shape']
        self.regularizer = params['regularizer']
        self.dropout = params['dropout']
        self.batch_norm = params['batch_norm']
        self.activation = params['activation']
        self.loss = params['loss']
        self.optimizer = params['optimizer']
        self.metrics = params['metrics']
        self.output_shape = params['output_shape']
        self.mode = params['mode']
        self.save_dir = params['save_dir']
        self.scaler = params['scaler']
        self.train_time = None
        self.layers = []

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
            self.model.add(self.activation)


class ModularNetwork(Network):
    def __init__(self, params):
        super().__init__(params)

        self.input = Input(shape=self.input_shape)
        self.layers += [self.input]

        self.summary = None
        self.predict = None
        self.fit = None

    def add_layer(self, layer, arch=[]):
        if layer in main_layers:
            for neurons in arch:
                if layer == 'Dense':
                    self.layers += [Dense(neurons, kernel_regularizer=self.regularizer)(self.layers[-1])]

                if layer == 'Conv1D':
                    self.layers += [Conv1D(neurons[0], neurons[1], neurons[2], padding=self.padding,
                                           kernel_regularizer=self.regularizer)(self.layers[-1])]
                if layer == 'Conv2D':
                    self.layers += [Conv2D(neurons[0], neurons[1], neurons[2], padding=self.padding,
                                           kernel_regularizer=self.regularizer)(self.layers[-1])]
                if layer == 'Conv2DTranspose':
                    self.layers += [Conv2DTranspose(neurons[0], neurons[1], neurons[2], padding='same',
                                                    kernel_regularizer=self.regularizer)(self.layers[-1])]
                if layer == 'LSTM':
                    pass

                if self.dropout:
                    self.add_layer('Dropout')
                if self.batch_norm:
                    self.add_layer('BatchNormalization')
                self.add_activation()
        if layer == 'Dropout':
            self.layers += [Dropout(self.dropout)(self.layers[-1])]
        if layer == 'BatchNormalization':
            self.layers += [BatchNormalization(momentum=0.8)(self.layers[-1])]
        if layer == 'Reshape':
            self.layers += [Reshape(arch)(self.layers[-1])]
        if layer == 'Flatten':
            self.layers += [Flatten()(self.layers[-1])]

    def add_activation(self):
        if type(self.activation) == str:
            self.layers += [Activation(self.activation)(self.layers[-1])]
        else:
            self.layers += [self.activation(self.layers[-1])]

    def get_layer(self, layer=None, pos=0):
        if layer is None:
            return self.layers[pos]
        l = [x for x in self.model.layers if layer == x._name]
        print(l)
        l = [self.model.layers.index(i) for i in l]
        print(l)
        return self.layers[l[pos]+1]

    def link_model_methods(self):
        self.summary = self.model.summary
        self.predict = self.model.predict
        self.fit     = self.model.fit

    def compile(self):
        self.output = Activation(act_dict[self.mode])(self.layers[-1])
        self.model = Model(self.input, self.output)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.link_model_methods()


class FCNN(SequentialNetwork):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']

        self.__compile()

    def __compile(self):  # NN build
        self.model.add(Dense(self.architecture[0], kernel_regularizer=self.regularizer, input_shape=self.input_shape))
        if self.dropout != 0:
            self.model.add(Dropout(self.dropout))
        if self.batch_norm:
            self.model.add(BatchNormalization())
        self.add_activation()
        for i in self.architecture[1:]:
            self.model.add(Dense(i, kernel_regularizer=self.regularizer))
            if self.dropout != 0:
                self.model.add(Dropout(self.dropout))
            if self.batch_norm:
                self.model.add(BatchNormalization())
            self.add_activation()

        self.model.add(Dense(self.output_shape))
        self.model.add(Activation(act_dict[self.mode]))

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if self.save_dir:
            self.save()


class CNN2D(ModularNetwork):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']
        self.archNN = params['archNN']
        self.padding = params['padding']

        self.build_model()
        self.compile()

    def build_model(self):
        self.add_layer('Conv2D', self.architecture)
        self.add_layer('Flatten')
        self.add_layer('Dense', self.archNN)
        self.add_layer('Dense', self.output_shape)

