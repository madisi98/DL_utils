from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('DL_utils/')
from networks import *


class Generator(ModularNetwork):
    def __init__(self, params):
        super().__init__(params)

        self.latent_dim   = params['input_shape']
        self.architecture = params['architecture']

        self.__compile()

    def __compile(self):
        aux_layer = self.input
        for i in self.architecture:
            aux_layer = Dense(i, kernel_regularizer=self.regularizer)(aux_layer)
            if self.batch_norm:
                aux_layer = BatchNormalization(momentum=0.8)
            if self.dropout:
                aux_layer = Dropout(self.dropout)(aux_layer)
            aux_layer = self.add_activation(aux_layer)
        aux_layer = Dense(np.prod(self.output), kernel_regularizer=self.regularizer)(aux_layer)

        self.output = Activation(act_dict[self.mode])(aux_layer)
        self.model  = Model(self.input, self.output)

    def generate_fake_samples(self, n_samples):
        latent_z = np.random.uniform(0, 1, [n_samples, self.latent_dim])
        samples = self.model.predict(latent_z)
        zeros = np.zeros([n_samples, 1])
        return samples, zeros


class Discriminator(ModularNetwork):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']

        self.__compile()

    def __compile(self):
        aux_layer = self.input
        for i in self.architecture:
            aux_layer = Dense(i, kernel_regularizer=self.regularizer)(aux_layer)
            if self.batch_norm:
                aux_layer = BatchNormalization(momentum=0.8)
            if self.dropout:
                aux_layer = Dropout(self.dropout)(aux_layer)
            aux_layer = self.add_activation(aux_layer)

        self.d_output = Dense(self.output[0], kernel_regularizer=self.regularizer)(aux_layer)
        self.d_output = Activation(act_dict[self.mode[0]])(self.d_output)
        self.model = Model(self.input, self.d_output)
        self.model.compile(loss=self.loss[0], optimizer=self.optimizer[0], metrics=self.metrics)

        self.c_output = Dense(self.output[1])(aux_layer)
        self.c_output = Activation(act_dict[self.mode[1]])(self.c_output)
        self.c_model = Model(self.input, self.c_output)
        self.c_model.compile(loss=self.loss[1], optimizer=self.optimizer[1], metrics=self.metrics)


class GAN:
    def __init__(self, d_params, g_params):
        self.discriminator = None
        self.generator     = None
        self.classifier    = None
        self.model         = None
        
        self.define_discriminator(d_params) # Compiles a Classifier too
        self.define_generator(g_params)
        self.define_gan()
    
    def define_discriminator(self, params):
        self.discriminator = Discriminator(params)
        self.classifier = Network(params)
        self.classifier.model = self.discriminator.c_model
        
    def define_generator(self, params):
        self.generator = Generator(params)

    def define_gan(self):
        self.discriminator.model.trainable = False # Freeze Discriminator weights
        out = self.discriminator.model(self.generator.output)
        self.model = Model(self.generator.input, out) # Generator input and Discriminator output linked
        self.model.compile(loss=self.discriminator.loss[0], optimizer=self.discriminator.optimizer[0]) # Discriminator Values




