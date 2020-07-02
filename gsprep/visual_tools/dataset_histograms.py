import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def plot_per_sample_histograms(data, ids = None , log = False , bins = 50, title = None, show=False, figsize=(10,10)):
    """
    Plot data histogram for every sample in dataset
    :param data: (n_samples, x, y, z)
    :param ids: sample ids
    :param log: boolean, plot log scale
    :param bins: int, number of histogram bins
    :param title: plot title
    :param show: show plot
    :param figsize: size of plot figure
    :return: figure
    """
    n_cols = 3
    n_rows = int(data.shape[0] / n_cols) + 1
    fig = plt.figure(figsize=figsize)
    for sample_index in range(data.shape[0]):
        plot_index = sample_index + 1
        plt.subplot(n_rows, n_cols, plot_index)
        plt.hist(data[sample_index].reshape((-1, )), bins=bins, log=log)
        sub_title = str(sample_index)
        if ids is not None:
            sub_title += '_' + ids[sample_index]
        plt.title(sub_title)
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show:
        plt.show()
    return fig

def plot_dataset_histogram(data, num_positions=100, log=False, mask_zero=False, dataset_label=None, alpha=0.05, color=None,
                           axis=None, figsize=(10,10)):
    """
    Plot an histogram for a dataset
    :param data: (n_samples, x, y, z)
    :param num_positions: number of position sampled in kernel
    :param log: display in log scale
    :param mask_zero: mask zeros out of data
    :param dataset_label: label for all values in this dataset
    :param alpha:
    :param color:
    :param axis:
    :param figsize:
    :return:
    """
    if axis is None:
        fig = plt.figure(figsize=figsize)
        axis = fig.add_subplot(1, 1, 1)
    for subj_idx in range(data.shape[0]):
        label = None
        if dataset_label is not None and subj_idx == 0:
            label = dataset_label
        plot_histogram(data[subj_idx], num_positions=num_positions, log=log, mask_zero=mask_zero, label=label,
                       alpha=alpha, color=color, axis=axis)


def plot_histogram(data, num_positions=100, log=False, mask_zero=False, label=None, alpha=0.05, color=None, axis=None,
                   figsize=(10,10)):
    """
    Plot an histogram for this data sample
    :param data: (n_samples, x, y, z)
    :param num_positions: number of position sampled in kernel
    :param log: display in log scale
    :param mask_zero: mask zeros out of data
    :param label:
    :param alpha:
    :param color:
    :param axis:
    :param figsize:
    :return:
    """
    if axis is None:
        fig = plt.figure(figsize=figsize)
        axis = fig.add_subplot(1, 1, 1)

    values = data.ravel()
    if mask_zero:
        values = values[np.argwhere(values != 0)].ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)
    if log:
        axis.set_yscale("log")

