import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import metrics
from tensorflow.keras.backend import epsilon


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


def mean_absolute_percentage_error(y_true, y_pred):
    """ MAPE metric

    Computes mean absolute percentage error, a metric for regression problems
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def recall(y_true, y_pred):
    """ Recall metric.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    recall = 0
    for i in range(y_pred.shape[1]):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true[:, i] * y_pred[:, i], 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true[:, i], 0, 1)))
        recall += true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall / tf.cast(y_pred.shape[1], np.float32)


class Precision(tf.keras.metrics.Metric):

    def __init__(self, name='precision', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.classes = 0
        self.true_positives = 0.
        self.predicted_positives = 0.

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.classes == 0:
            self.classes = y_true.shape[1]
            self.true_positives = [0. for x in range(self.classes)]
            self.predicted_positives = [0. for x in range(self.classes)]
        for i in range(self.classes):
            self.true_positives[i] += tf.reduce_sum(tf.round(tf.clip_by_value(y_true[:, i] * y_pred[:, i], 0, 1)))
            self.predicted_positives[i] += tf.reduce_sum(tf.round(tf.clip_by_value(y_pred[:, i], 0, 1)))

    def result(self):
        return tf.reduce_mean(self.true_positives / (self.predicted_positives + epsilon))

    def reset_states(self):
        self.true_positives = [0. for x in range(self.classes)]
        self.predicted_positives = [0. for x in range(self.classes)]



def precision(y_true, y_pred):
    """ Precision metric.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    precision = 0
    for i in range(y_pred.shape[1]):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true[:, i] * y_pred[:, i], 0, 1)))
        predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred[:, i], 0, 1)))
        precision += true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision / tf.cast(y_pred.shape[1], np.float32)


def f1_score(y_true, y_pred):
    """F1 score metric.

    Computes the precision, a metric for multi-label classification.
    It equals the harmonic mean of precision and recall.
    """
    f1_score = 0
    for i in range(y_pred.shape[1]):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true[:, i] * y_pred[:, i], 0, 1)))
        predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred[:, i], 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true[:, i], 0, 1)))

        p = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        r = true_positives / (possible_positives + tf.keras.backend.epsilon())
        f1_score += 2*((p*r)/(p+r+tf.keras.backend.epsilon()))

    return f1_score / tf.cast(y_pred.shape[1], np.float32)


def get_metrics(y, preds, mode):
    metric = '\n\n'
    metric += 'Trues shape:        ' + str(y.shape) + '\n'
    metric += 'Predictions shape:  ' + str(preds.shape) + '\n'

    if mode == 0 or mode == 2:  # Classification or binary
        if mode == 0:
            preds = np.argmax(preds, 1)
            y = np.argmax(y, 1)
        else:
            preds = np.round(preds)

        metric += 'True values:        ' + str(np.unique(y)) + '\n'
        metric += 'Prediction values:  ' + str(np.unique(preds)) + '\n'

        df = pd.DataFrame({
            'f1_score': metrics.f1_score(y, preds, average=None),
            'precision': metrics.precision_score(y, preds, average=None),
            'recall': metrics.recall_score(y, preds, average=None),
        }, index=list(range(np.unique(y).shape[0]))).T
        df['Macro'] = np.mean(df, axis=1)
        df['Micro'] = pd.Series({
            'f1_score': metrics.f1_score(y, preds, average='micro'),
            'precision': metrics.precision_score(y, preds, average='micro'),
            'recall': metrics.recall_score(y, preds, average='micro'),
        })

        metric += '\n\n' + str(df.T) + '\n\n'

        metric += 'Confusion matrix:' + '\n'
        metric += str(metrics.confusion_matrix(y, preds))

    elif mode == 1:  # Regression
        df = pd.Series({
            'mean_squared_error': metrics.mean_squared_error(y, preds),
            'mean_absolute_error': metrics.mean_absolute_error(y, preds),
            # 'mean_squared_logarithmic_error': metrics.mean_squared_log_error(y, preds),
            'mean_absolute_percentage_error': mean_absolute_percentage_error(y, preds),
            'r2_score': metrics.r2_score(y, preds),
        })
        metric += '\n\n' + str(df) + '\n\n'

    return metric, df

