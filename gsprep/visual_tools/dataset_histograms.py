import matplotlib.pyplot as plt

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