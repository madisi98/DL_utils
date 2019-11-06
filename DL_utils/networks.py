from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import *
from DL_utils.utils import *
import time


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

    def trainNetwork(self, params, X_train, Y_train, X_val, Y_val, X_test=None, Y_test=None):
        max_steps = params['max_epochs']
        max_idle = params['max_idle']
        batch_size = params['batch_size']
        verbose = params['verbose']
        epochs_per_steps = params['epochs_per_step']
        class_weights = params['class_weights']
        es_delta = params['es_delta']
        es_target = params['es_target']

        step_count = 0
        idle = 0
        time_start = time.time()
        val_metrics = {x: '---' for x in self.metrics}
        test_metrics = {x: '---' for x in self.metrics}
        if es_target == 'loss':
            max_val_metric = starting_metric[self.loss]
        else:
            max_val_metric = starting_metric[es_target]

        if verbose:
            print("Started training:\n")
        while self.keep_training(step_count, idle, val_metrics[es_target], max_steps, max_idle):
            with tf.device('/gpu:0'):
                step_time = time.time()
                self.model.fit(X_train, Y_train, epochs=epochs_per_steps, batch_size=batch_size,
                               validation_data=(X_val, Y_val), verbose=verbose > 2, shuffle=True, class_weight=class_weights)

                # Metrics after current step
                train_metrics = self.evaluate(X_train, Y_train, verbose=verbose)
                val_metrics = self.evaluate(X_val, Y_val, verbose=verbose)
                if X_test is not None:
                    test_metrics = self.evaluate(X_test, Y_test, verbose=verbose)

                if self.check_idle(val_metrics[es_target], max_val_metric, es_target, es_delta):
                    max_val_metric = val_metrics[es_target]
                    idle = 0
                    if self.save_dir:
                        self.save()
                else:
                    idle += 1
                step_count += 1

                if verbose:
                    print('Step {} metrics:'.format(step_count))
                    print(pd.DataFrame([train_metrics, val_metrics, test_metrics], columns=train_metrics.keys(), index=['Train', 'Val', 'Test']).T)
                    print('Exec. time: {} Idle: {}\n'.format(int(time.time() - step_time), idle))

        self.train_time = time.time() - time_start

    def check_idle(self, current, best, es_target=None, es_delta=10e-6):
        diff = 0
        if es_target == 'loss':
            es_target = self.loss
        if starting_metric[es_target] == 0:
            diff = current - best
        elif starting_metric[es_target] == np.inf:
            diff = best - current
        return diff > es_delta

    def keep_training(self, s, i, l, max_steps, max_idle):
        if s > 1 and np.isnan(l):  # If gradient made loss explode
            return False
        if s >= max_steps or i >= max_idle:  # Normal early stopping of a net (it has converged)
            return False
        return True

    def evaluate(self, X, Y, verbose=0):
        metrics = {x: 0 for x in self.metrics}
        m = self.model.evaluate(X, Y, batch_size=2**30, verbose=verbose > 4)
        metrics['loss'] = m[0]
        for i, j in enumerate(self.metrics[:-1]):
            metrics[j] = m[i+1]
        return metrics

    def pred(self, X_test, Y_test, show_mode=None):
        self.load()
        preds = self.model.predict(X_test, batch_size=2**30)
        metrics, df_metrics = get_metrics(Y_test, preds, self.mode)

        if show_mode == 'file' or show_mode == 'both':
            if self.save_dir:
                with open(self.save_dir + '.txt', 'w') as f:
                    self.model.summary(print_fn=lambda x: f.write(x + '\n'))
                    f.write(metrics)
            else:
                print('Can\'t open file, no save dir specified')

        elif show_mode == 'print' or show_mode == 'both' or show_mode == 'verbose':
            self.model.summary()
            print(metrics)

        return preds, df_metrics
    
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


class ModularNetwork(Network):
    def __init__(self, params):
        super().__init__(params)

        self.input = Input(shape=self.input_shape)
        self.output = None
        self.layers += [self.input]

        self.summary = None
        self.predict = None
        self.fit = None

        self.check_attrs()

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

    def check_attrs(self):
        if type(self.output_shape) == int:
            self.output_shape = [self.output_shape, ]

        self.metrics += ['loss']

    def compile(self):
        self.output = Activation(act_dict[self.mode])(self.layers[-1])
        self.model = Model(self.input, self.output)

        m = list(pd.Series(self.metrics[:-1]).replace(custom_objects))

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=m)
        self.link_model_methods()


class FCNN(ModularNetwork):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']

        self.build_model()
        self.compile()

    def build_model(self):
        self.add_layer('Dense', self.architecture)
        self.add_layer('Dense', self.output_shape)


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

