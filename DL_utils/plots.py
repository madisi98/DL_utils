import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics


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


def corr_matrix(df, figsize=(10, 10)):
    """ Function that colors a correlation matrix of a given DataFrame or np.array
    :param df: Data to be analyzed
    :param figsize: Matplotlib figure size
    :return: A plot of the matrix
    """

    df = pd.DataFrame(df)
    corr = df.corr()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(df.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.show()


def roc_curve(y_true, y_pred, single_plot=False, figsize=None):
    """ Computes and plots ROC curves for a given dataset

    :param y_true: Original labels - Must be one-hot
    :param y_pred: Predicted labels - Output from model.predict is valid
    :param single_plot: Wheter you want a single plot or
                        one for each class plus macro and micro plots
    :param figsize: Matplotlib figure size
    :return: None
    """
    rows = int((y_true.shape[1] + 3) / 2)
    columns = 2
    lw = 2
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(y_true.shape[1]):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    fpr["macro"] = np.unique(np.concatenate([fpr[i] for i in range(y_true.shape[1])]))
    mean_tpr = np.zeros_like(fpr['macro'])
    for i in range(y_true.shape[1]):
        mean_tpr += scipy.interp(fpr['macro'], fpr[i], tpr[i])
    tpr["macro"] = mean_tpr / y_true.shape[1]
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    if not figsize:
        if single_plot:
            figsize = (7,7)
        else:
            figsize = (12, 5*rows)

    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    for i, j in enumerate(fpr.keys()):
        c = '' if single_plot else ': Class {}'.format(j)
        if not single_plot:
            plt.subplot(rows, columns, i + 1)
            plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.plot(fpr[j], tpr[j], lw=lw, label='ROC curve {} (area = {:.2f})'.format(j, roc_auc[j]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC curve{}'.format(c), fontsize=20)
        if single_plot:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
        else:
            plt.legend(loc='lower right', fontsize=14)
            plt.tight_layout()
    plt.show()

