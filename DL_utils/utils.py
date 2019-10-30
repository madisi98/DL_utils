import numpy as np
import tensorflow as tf
import os
import time

from matplotlib import pyplot as plt
from sklearn import metrics

main_layers = ['Dense', 'Conv1D', 'Conv2D', 'LSTM', 'Conv2DTranspose']

act_dict = {
    0: 'softmax', # Multi-class classification
    1: 'linear',  # Regression
    2: 'sigmoid', # Binary Classification
    3: 'tanh',    # Generative
}

starting_metric = {
    'f1_score':                 0,
    'acc':                      0,
    'accuracy':                 0,
    'binary_crossentropy':      np.inf,
    'categorical_crossentropy': np.inf,
    'mean_squared_error':       np.inf,
    'mean_absolute_error':      np.inf,
    'standard':                 np.inf,
    'cuadratic':                np.inf,
}


def compare_metrics(current, best, metric):
    if starting_metric[metric] == 0:
        return current > best
    elif starting_metric[metric] == np.inf:
        return current < best


def create_results_directory(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # os.makedirs(save_dir + '/models'.format(save_dir))
        # os.makedirs(save_dir + '/metrics'.format(save_dir))

def corr_matrix(df):
    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data1.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data1.columns)
    ax.set_yticklabels(data1.columns)
    #Descomentar linea para guardar el plot como .png, el primer campo es el nombre del archivo resultante
    save_file = 'corr_matrix_{}'.format('_'.join(file1.split('/')))
    save_file = '_'.join(save_file.split('.'))
    plt.savefig(save_file + '.png', bbox_inches='tight')
    plt.show()
        
        
def analyze(X, verbose=True, plot=True):
    results = {
        'shape': X.shape,
        'mean':  np.mean(X),
        'std':   np.std(X),
        'var':   np.var(X),
        'max':   np.max(X),
        'min':   np.min(X)
    }

    if verbose:
        for i in results.keys():
            print('X\'s {} is: {}'.format(i, results[i]))
    if plot:
        start = 0
        end = len(X)
        xaxis = np.arange(start, end)
        trues_line = plt.plot(xaxis, X[start:end], label='values')
        mean_line = plt.plot(
            xaxis, [results['mean']] * (end - start), label='mean')
        std_line11 = plt.plot(
            xaxis, [results['mean'] + results['std']] * (end - start), label='std', color='red')
        std_line12 = plt.plot(
            xaxis, [results['mean'] - results['std']] * (end - start), label='std', color='red')
        std_line21 = plt.plot(xaxis, [results['mean'] + results['std'] * 3] * (
                end - start), label='std*3', color='green', linestyle='--')
        std_line22 = plt.plot(xaxis, [results['mean'] - results['std'] * 3] * (
                end - start), label='std*3', color='green', linestyle='--')
        plt.legend()
        plt.show()
    return results


def f1_score(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.argmax(y_true, axis=1)

    tp = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.int32))
    fp = tf.shape(y_pred) - tp
    fn = fp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * (precision * recall) / (precision + recall)


def keep_training(s, i, l, max_steps, max_idle):
    if s > 1 and np.isnan(l):  # If gradient made loss explode
        return False
    if s >= max_steps or i >= max_idle:  # Normal early stopping of a net (it has converged)
        return False
    return True


def get_metrics(y, preds, mode):
    metric = ''
    metric += 'Trues shape:        ' + str(y.shape) + '\n'
    metric += 'Predictions shape:  ' + str(preds.shape) + '\n'
    pred_probs = preds

    if mode == 0 or mode == 2:  # Classification or binary
        if mode == 0:
            preds = np.argmax(preds, 1)
            y = np.argmax(y, 1)
        else:
            preds = np.round(preds)

        metric += 'True values:        ' + str(np.unique(y)) + '\n'
        metric += 'Prediction values:  ' + str(np.unique(preds)) + '\n'

        metric += '\nF1, precision, recall (macro/micro):' + '\n'
        metric += str(metrics.f1_score(y, preds, average='macro')) + '\n'
        metric += str(metrics.f1_score(y, preds, average='micro')) + '\n'
        metric += str(metrics.precision_score(y, preds, average='macro')) + '\n'
        metric += str(metrics.precision_score(y, preds, average='micro')) + '\n'
        metric += str(metrics.recall_score(y, preds, average='macro')) + '\n'
        metric += str(metrics.recall_score(y, preds, average='micro')) + '\n\n'

        metric += 'Confusion matrix:' + '\n'
        metric += str(metrics.confusion_matrix(y, preds))

        score = metrics.precision_score(y, preds, average='micro')

    elif mode == 1:  # Regression

        metric += '\nMSE, MAE, MAPE, R2:' + '\n'
        metric += str(metrics.mean_squared_error(y, preds)) + '\n'
        metric += str(metrics.mean_absolute_error(y, preds)) + '\n'
        metric += str(np.mean(np.abs((y - preds) / y)) * 100) + '\n'  # MAPE
        metric += str(metrics.r2_score(y, preds)) + '\n'

    return pred_probs, metric, score


def subsample_dataset(X, Y, samples_per_class=None, proportions=None):
    """Returns a sub-sample a given dataset.

    If the argument 'mode' isn't passed in, the default mode 'balanced' is used.

    Parameters
    ----------
    X : np.array or pd.DataFrame, 
        Dataset to be sub-sampled.
        
    Y : np.array or pd.DataFrame, 
        Tags for X.
        
    samples_per_class : int or list of ints, [Must be greater than 0]
                        Number of samples per class.
        
    proportions : float or list of floats, [Must be between 0 and 1]
                  Proportion of the original dataset wanted to be kept.

    """
    X_classes, Y_classes = [], []
    if proportions is not None:
        samples_per_class = []
        if type(proportions) != list:
            proportions = [proportions]*Y.shape[1]
        for i in range(Y.shape[1]):
            samples = Y[Y[:,i] == 1].shape[0]
            samples_per_class += [int(samples*proportions[i])]
                
    if samples_per_class is not None:
        if type(samples_per_class) != list:
            samples_per_class = [samples_per_class]*Y.shape[1]
        for i in range(Y.shape[1]):
            class_x = X[Y[:,i] == 1]
            class_y = Y[Y[:,i] == 1]
            random_sample = np.random.choice(range(class_x.shape[0]), samples_per_class[i], replace=False)
            if not i:
                X_classes, Y_classes = class_x[random_sample], class_y[random_sample]
            else:
                X_classes = np.append(X_classes, class_x[random_sample], axis=0)
                Y_classes = np.append(Y_classes, class_y[random_sample], axis=0)
    
    else: return X,Y
    return X_classes, Y_classes

####################
## LOSS FUNCTIONS ##
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
    'f1_score':f1_score,
    'feature_matching_loss':feature_matching_loss,
    'binary_crossentropy_nolog':binary_crossentropy_nolog,
    'categorical_crossentropy_nolog':categorical_crossentropy_nolog,
}