import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
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


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


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


def compute_metrics(y, preds, mode, labels=None):
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

        if labels is None:
            labels = list(range(np.unique(y).shape[0]))

        df = pd.DataFrame({
            'f1_score': metrics.f1_score(y, preds, average=None),
            'precision': metrics.precision_score(y, preds, average=None),
            'recall': metrics.recall_score(y, preds, average=None),
        }, index=labels).T
        df['Macro'] = np.mean(df, axis=1)
        df['Micro'] = pd.Series({
            'f1_score': metrics.f1_score(y, preds, average='micro'),
            'precision': metrics.precision_score(y, preds, average='micro'),
            'recall': metrics.recall_score(y, preds, average='micro'),
        })

        matrix = pd.DataFrame(metrics.confusion_matrix(y, preds), columns=labels, index=labels)

        metric += '\n\n' + str(df.T) + '\n\n'

        metric += 'Confusion matrix:' + '\n'
        metric += str()

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

