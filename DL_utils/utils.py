import os
from .plots import *
from .metrics import *

main_layers = ['Dense', 'Conv1D', 'Conv2D', 'LSTM', 'Conv2DTranspose']

act_dict = {
    0: 'softmax',  # Multi-class classification
    1: 'linear',   # Regression
    2: 'sigmoid',  # Binary Classification
    3: 'tanh',     # Generative
}


def create_results_directory(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # os.makedirs(save_dir + '/models'.format(save_dir))
        # os.makedirs(save_dir + '/metrics'.format(save_dir))


def subsample_dataset(X, Y, samples_per_class=None, proportions=None):
    """Returns a sub-sample a given dataset.

    If the argument 'mode' isn't passed in, the default mode 'balanced' is used.

    :param X : np.array or pd.DataFrame,
               Dataset to be sub-sampled.
    :param Y : np.array or pd.DataFrame,
               Tags for X.
    :param samples_per_class : int or list of ints, [Must be greater than 0]
                               Number of samples per class.
    :param proportions : float or list of floats, [Must be between 0 and 1]
                         Proportion of the original dataset wanted to be kept.

    :return: np.array with the random sub-sample
    """
    X_classes, Y_classes = [], []
    if proportions is not None:
        samples_per_class = []
        if type(proportions) != list:
            proportions = [proportions] * Y.shape[1]
        for i in range(Y.shape[1]):
            samples = Y[Y[:, i] == 1].shape[0]
            samples_per_class += [int(samples * proportions[i])]

    if samples_per_class is not None:
        if type(samples_per_class) != list:
            samples_per_class = [samples_per_class] * Y.shape[1]
        for i in range(Y.shape[1]):
            class_x = X[Y[:, i] == 1]
            class_y = Y[Y[:, i] == 1]
            random_sample = np.random.choice(range(class_x.shape[0]), samples_per_class[i], replace=False)
            if not i:
                X_classes, Y_classes = class_x[random_sample], class_y[random_sample]
            else:
                X_classes = np.append(X_classes, class_x[random_sample], axis=0)
                Y_classes = np.append(Y_classes, class_y[random_sample], axis=0)

    else:
        return X, Y
    return X_classes, Y_classes


####################
#  LOSS FUNCTIONS  #
####################


def feature_matching_loss(y_true, y_pred):
    means = tf.square( tf.reduce_mean(y_true, axis=1) - tf.reduce_mean(y_pred, axis=1) )
    stds = tf.square( tf.math.reduce_std(y_true, axis=1) - tf.math.reduce_std(y_pred, axis=1) )
    return tf.reduce_mean( tf.concat( [means, stds], axis=0 ) )


def binary_crossentropy_nolog(Y_true, Y_pred):
    return - tf.reduce_mean( Y_true * Y_pred + (1-Y_true) * (1-Y_pred) ) + 1


def categorical_crossentropy_nolog(Y_true, Y_pred):
    return - tf.reduce_mean( tf.reduce_sum(Y_true * Y_pred, axis=1) ) + 1


custom_objects = {
    'feature_matching_loss': feature_matching_loss,
    'binary_crossentropy_nolog': binary_crossentropy_nolog,
    'categorical_crossentropy_nolog': categorical_crossentropy_nolog,
    'f1_score': f1_score,
    'precision': precision,
    'recall': recall,
}