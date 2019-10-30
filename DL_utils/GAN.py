from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import load_model

import sys
sys.path.append('DL_utils/')
from networks import *


class Generator(ModularNetwork):
    def __init__(self, params):
        super().__init__(params)

        self.latent_dim   = params['input_shape']
        self.architecture = params['architecture']
        self.padding = 'same'

        self.__compile()

    def __compile(self):
        for layer, arch in self.architecture:
            self.add_layer(layer, arch)

        if len(self.output_shape) == 1:
            out_layer = Dense(self.output_shape, kernel_regularizer=self.regularizer)(self.layers[-1])
        else:
            out_layer = Conv2D(1, self.architecture[1][1][:2], padding='same',
                               kernel_regularizer=self.regularizer)(self.layers[-1])

        self.output = Activation(act_dict[self.mode])(out_layer)
        self.model = Model(self.input, self.output)

        self.link_model_methods()

    def generate_latent_vectors(self, n):
        return np.random.uniform(0, 1, [n, self.latent_dim])

    def generate_fake_samples(self, n_samples, latent_z=None):
        if latent_z is None:
            latent_z = self.generate_latent_vectors(n_samples)
        x = self.model.predict(latent_z, batch_size = 4096)
        return x

    def generate_fake_dataset(self, n_samples, classes):
        x = self.generate_fake_samples(n_samples)
        y_disc = np.zeros([n_samples, 1])
        y_class = np.zeros([n_samples, classes])
        y_class[:, -1] = 1
        return x, y_disc, y_class


class Discriminator(ModularNetwork):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']
        self.padding = 'same'

        self.__compile()

    def __compile(self):
        for layer, arch in self.architecture:
            self.add_layer(layer, arch)

        self.last_hidden_model = Model(self.input, self.layers[-1])
        
        self.d_output = Dense(self.output_shape[0], kernel_regularizer=self.regularizer)(self.layers[-1])
        self.d_output = Activation(act_dict[self.mode[0]])(self.d_output)
        self.model = Model(self.input, self.d_output)
        self.model.compile(loss=self.loss[0], optimizer=self.optimizer[0], metrics=self.metrics)

        self.model.trainable = False

        self.layers += [Dense(100, kernel_regularizer=self.regularizer)(self.get_layer('flatten'))]
        self.c_output = Dense(self.output_shape[1])(self.layers[-1])
        self.c_output = Activation(act_dict[self.mode[1]])(self.c_output)
        self.c_model = Model(self.input, self.c_output)
        self.c_model.compile(loss=self.loss[1], optimizer=self.optimizer[1], metrics=self.metrics)


class GAN:
    def __init__(self, d_params, g_params, GAN_params):
        self.save_dir = GAN_params['save_dir']
        self.hidden_layer_output = GAN_params['hidden_layer_output']
        self.batch_size = GAN_params['batch_size']

        self.discriminator = None
        self.generator     = None
        self.classifier    = None
        self.model         = None
        
        self.define_discriminator(d_params) # Compiles a Classifier too
        self.define_generator(g_params)
        self.define_gan()

        self.latent_vectors = self.generator.generate_latent_vectors(100)
    
    def define_discriminator(self, params):
        # Defines Discriminator and Classifier
        self.discriminator = Discriminator(params)
        self.classifier = Network(params)
        self.classifier.model = self.discriminator.c_model
        self.discriminator.mode, self.classifier.mode = self.discriminator.mode
        self.discriminator.save_dir, self.classifier.save_dir = self.discriminator.save_dir

        self.discriminator.link_model_methods()

    def define_generator(self, params):
        # Defines generator
        self.generator = Generator(params)

    def define_gan(self):
        # Compiles the GAN model (for training generator weights)
        self.discriminator.model.trainable = False # Freeze Discriminator weights
        if not self.hidden_layer_output:
            out = self.discriminator.model(self.generator.output)
        else:
            out = Model(self.discriminator.input, self.discriminator.layers[-1])(self.generator.output)
        self.model = Model(self.generator.input, out) # Generator input and Discriminator output linked
        self.model.compile(loss=self.generator.loss,
                           optimizer=self.generator.optimizer)
                           #metrics=self.discriminator.metrics) # Discriminator Values
            
    def save(self):
        # Checkpoints every model of the GAN
        self.discriminator.save()
        self.generator.save()
        self.classifier.save()
        self.model.save(self.save_dir + 'GAN.h5')
        
    def load(self):
        # Loads every model of the GAN (From save_dir)
        self.discriminator.load()
        self.generator.load()
        self.classifier.load()
        self.model = load_model(self.save_dir + 'GAN.h5', custom_objects=custom_objects)

    def check_idle(self, c, m, mean='standard', verbose=False):
        # Validation loss from (d)iscriminator, (g)enerator, (c)lassifier and (m) best values
        c, m = np.array(c), np.array(m)
        if verbose:
            print(c,m)
        if mean == 'standard':
            a = np.mean(c[[0,2]])
            b = np.mean(m[[0,2]])
        elif mean == 'cuadratic':
            a = np.sqrt(np.mean(np.square(c[[0,2]])))  # Squared mean; This penalizes the training if any net gets high losses.
            b = np.sqrt(np.mean(np.square(m[[0,2]])))
        elif mean == 'accuracy':
            if 0 in m:
                return True
            b = np.mean([c[1], 1/c[2]]) # Inverse order because higher is better
            a = np.mean([m[1], 1/m[2]])
        return a < b
    
    def get_losses(self, X_real, Y_real_disc, Y_real_class, X_class=None):
        X_fake, Y_fake_disc, Y_fake_class = self.generator.generate_fake_dataset(X_real.shape[0], Y_real_class.shape[1])

        if X_class is None:
            X_class = X_real
        X = np.append(X_real, X_fake, axis=0)
        latent_z = self.generator.generate_latent_vectors(X_real.shape[0])
        Y_disc = np.append(Y_real_disc, Y_fake_disc, axis=0)
        if not self.hidden_layer_output:
            Y_gen = np.ones(latent_z.shape[0])
        else:
            Y_gen = self.discriminator.last_hidden_model.predict(X_real)
        
        c_loss, c_acc = self.classifier.model.evaluate(X_class, Y_real_class, batch_size=self.batch_size, verbose=0)
        d_loss, d_acc = self.discriminator.model.evaluate(X, Y_disc, batch_size=self.batch_size, verbose=0)
        g_loss = self.model.evaluate(latent_z, Y_gen, batch_size=self.batch_size, verbose=0)
        
        return np.array([d_loss, d_acc, g_loss, c_loss, c_acc])

    def train(self, train_params, X_real, Y_real_disc, Y_real_class, X_real_val, Y_real_disc_val, Y_real_class_val):
        max_epochs = train_params['max_epochs']
        max_idle = train_params['max_idle']
        batch_size = train_params['batch_size']
        verbose = train_params['verbose']
        warm_up = train_params['warm_up']
        train_ratio = train_params['train_ratio']
        fake_ratio = train_params['fake_ratio']
        mean = train_params['mean']

        max_val_losses = np.array([starting_metric[mean]] * 7)
        val_loss = np.inf
        step_count = 0
        idle = 0
        time_start = time.time()

        if verbose:
            print("Started training:")

        while keep_training(step_count, idle, val_loss, max_epochs, max_idle):
            with tf.device('/gpu:0'):
                step_time = time.time()

                # Generate Dataset for current epoch
                X_fake, Y_fake_disc, Y_fake_class = self.generator.generate_fake_dataset(X_real.shape[0], Y_real_class.shape[1])
                X_fake_val, Y_fake_disc_val, Y_fake_class_val = self.generator.generate_fake_dataset(X_real_val.shape[0], Y_real_class_val.shape[1])

                # TRAIN DATASET
                X_train = np.append(X_real, X_fake, axis=0)
                latent_z = self.generator.generate_latent_vectors(int(fake_ratio*X_real.shape[0]))
                Y_train_disc = np.append(Y_real_disc, Y_fake_disc, axis=0)
                Y_train_class = np.append(Y_real_class, Y_fake_class, axis=0)
                if not self.hidden_layer_output:
                    Y_train_gen = np.ones(latent_z.shape[0])
                else:
                    Y_train_gen = self.discriminator.last_hidden_model.predict(fake_ratio*X_real)

                # VALIDATION DATASET
                X_val = np.append(X_real_val, X_fake_val, axis=0)
                latent_z_val = self.generator.generate_latent_vectors(X_real_val.shape[0])
                Y_val_disc = np.append(Y_real_disc_val, Y_fake_disc_val, axis=0)
                Y_val_class = np.append(Y_real_class_val, Y_fake_class_val, axis=0)
                if not self.hidden_layer_output:
                    Y_val_gen = np.ones(latent_z_val.shape[0])
                else:
                    Y_val_gen = self.discriminator.last_hidden_model.predict(X_real_val)

                # TRAIN
                self.discriminator.model.fit(X_train, Y_train_disc, epochs=train_ratio[1], batch_size=batch_size, validation_data=(X_val, Y_val_disc), shuffle=True, verbose=0)
                self.model.fit(latent_z, Y_train_gen, epochs=train_ratio[0], batch_size=batch_size, validation_data=(latent_z_val, Y_val_gen), shuffle=True, verbose=0)
                # class_hist = self.classifier.model.fit(X_train, Y_train_class, epochs=train_ratio[1], batch_size=batch_size, validation_data=(X_val, Y_val_class), shuffle=True, verbose=0)

                val_losses = self.get_losses(X_real_val, Y_real_disc_val, Y_real_class_val)

                step_count = step_count + 1
                if step_count >= warm_up:
                    if self.check_idle(val_losses, max_val_losses, mean):
                        max_val_losses = val_losses
                        idle = 0
                        self.save()
                    else:
                        idle = idle + 1
                step_time = np.round(time.time() - step_time)
                if verbose:
                    print('Step {:3d} idle {:2d}: Disc[{:.5f};{:.5f};{:.5f};{:.5f}]; Gen[{:.5f}]; Time: {:.2f}s'.format(
                           step_count, idle, *val_losses[:-2], step_time))

    def train_on_batches(self, train_params, X_real, X_real_class, Y_real_disc, Y_real_class, X_real_val, Y_real_disc_val, Y_real_class_val):
        max_epochs = train_params['max_epochs']
        max_idle = train_params['max_idle']
        verbose = train_params['verbose']
        plot = train_params['plot']
        warm_up = train_params['warm_up']
        train_ratio = train_params['train_ratio']
        fake_ratio = train_params['fake_ratio']
        mean = train_params['mean']
        train_classifier = train_params['train_classifier']

        step_count = 0
        epoch = 0
        idle = 0
        max_val_losses = [starting_metric[mean]]*7
        step_time = time.time()

        half_batch = int(self.batch_size / 2)
        batch_per_epoch = int(X_real.shape[0]/self.batch_size)
        n_steps = batch_per_epoch * max_epochs
        print('Epochs: {}; Batch Size: {}; Batch/Epoch: {}; Total Batches: {}'.format(max_epochs, self.batch_size, batch_per_epoch, n_steps))
        print('Started training on batches...\n')
        print('                        D Loss  D Acc           G Loss         C Loss   C acc                ')
        print('---------------------------------------------------------------------------------------------')
        while keep_training(epoch, idle, 0, max_epochs, max_idle):
            step_count += 1
            batch_index = np.random.choice(range(X_real.shape[0]), int(half_batch), replace=False)
            if half_batch > X_real_class.shape[0]:
                batch_index_class = np.arange(X_real_class.shape[0])
            else:
                batch_index_class = np.random.choice(range(X_real_class.shape[0]), int(half_batch), replace=False)
            X_fake, Y_fake_disc = self.generator.generate_fake_samples(half_batch), np.zeros(half_batch)
            latent_z = self.generator.generate_latent_vectors(int(fake_ratio * self.batch_size))
            if not self.hidden_layer_output:
                Y_train_gen = np.ones(latent_z.shape[0])
            else:
                Y_train_gen = self.discriminator.last_hidden_model.predict(X_real[batch_index])

            if train_classifier:
                c_loss, c_acc = self.classifier.model.train_on_batch(X_real_class[batch_index_class], Y_real_class[batch_index_class])
            d_loss1, d_acc1 = self.discriminator.model.train_on_batch(X_real[batch_index], Y_real_disc[batch_index])
            d_loss2, d_acc2 = self.discriminator.model.train_on_batch(X_fake, Y_fake_disc)
            g_loss = self.model.train_on_batch(latent_z, Y_train_gen)

            if step_count % batch_per_epoch == 0:
                epoch += 1
                
                train_losses = self.get_losses(X_real[batch_index], Y_real_disc[batch_index], Y_real_class[batch_index_class], X_real_class[batch_index_class])
                val_losses = self.get_losses(X_real_val, Y_real_disc_val, Y_real_class_val)
                if epoch > warm_up:
                    if self.check_idle(val_losses, max_val_losses, mean=mean):
                        max_val_losses = val_losses
                        idle = 0
                        self.save()
                    else:
                        idle += 1
                step_time = np.round(time.time() - step_time)
                if verbose:
                    print('Epoch {:3d} idle {:2d}: Disc[{:.5f};{:.5f}]; Gen[{:.5f}]; Class[{:.5f};{:.5f}]; Time: {}s'.format(
                        epoch, idle, *train_losses, step_time))
                    print('              Val: Disc[{:.5f};{:.5f}]; Gen[{:.5f}]; Class[{:.5f};{:.5f}]'.format(*val_losses))
                if plot:
                    rows = 2
                    columns = 10

                    plt.figure(figsize=(15, 3))
                    for i in range(rows * columns):
                        plt.subplot(rows, columns, i + 1)
                        plt.axis('off')
                        rn = self.generator.generate_fake_samples(1, latent_z=self.latent_vectors[i].reshape(1,-1))
                        plt.imshow(rn.reshape(28, 28), cmap='gray')
                    plt.show()
                step_time = time.time()

    def score(self, X_real_test, Y_real_class_test, Y_real_disc_test, load_checkpoint=False, show_mode='print'):
        if load_checkpoint:
            self.load()

        X_fake_test, Y_fake_disc_test, Y_fake_class_test = self.generator.generate_fake_dataset(X_real_test.shape[0], Y_real_class_test.shape[1])
        X_test = np.append(X_real_test, X_fake_test, axis=0)
        Y_test_disc = np.append(Y_real_disc_test, Y_fake_disc_test, axis=0)
        Y_test_class = np.append(Y_real_class_test, Y_fake_class_test, axis=0)

        self.discriminator.pred(X_test, Y_test_disc, show_mode=show_mode)
        self.classifier.pred(X_real_test, Y_real_class_test, show_mode=show_mode)

