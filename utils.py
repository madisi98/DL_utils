import numpy as np
import tensorflow as tf
import os
import time

from matplotlib import pyplot as plt
from sklearn import metrics

act_dict = {
    0: 'softmax',
    1: 'linear'
}

starting_metric = {
    'f1_score':                 0,
    'acc':                      0,
    'accuracy':                 0,
    'categorical_crossentropy': np.inf,
    'mean_squared_error':       np.inf,
    'mean_absolute_error':      np.inf
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


def analyze(X, verbose=True, plot=True):
    results = {}
    results['shape'] = X.shape
    results['mean'] = np.mean(X)
    results['std'] = np.std(X)
    results['var'] = np.var(X)
    results['max'] = np.max(X)
    results['min'] = np.min(X)
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
        std_line21 = plt.plot(xaxis, [results['mean'] + results['std'] * 2.5] * (
                end - start), label='std*2', color='green', linestyle='--')
        std_line22 = plt.plot(xaxis, [results['mean'] - results['std'] * 2.5] * (
                end - start), label='std*2', color='green', linestyle='--')
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
    if s > 1 and (np.isnan(l) or np.isinf(l)):  # If gradient made loss explode
        return False
    if s >= max_steps or i >= max_idle:  # Normal early sttoping of a net (it has converged)
        return False
    return True


def get_metrics(y, preds, mode):
    metric = ''
    metric += 'Trues shape:        ' + str(y.shape) + '\n'
    metric += 'Predictions shape:  ' + str(preds.shape) + '\n'
    pred_probs = preds

    if mode == 0:  # Classification
        preds = np.argmax(preds, 1)
        y = np.argmax(y, 1)

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