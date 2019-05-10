import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_easy21_value(v, title='', iter=None):
    """

    :param v: value function calculated
    :param title: a string name
    :return: figure in png format
    """
    x_range = range(0, 21, 1)
    y_range = range(0, 10, 1)
    X, Y = np.meshgrid(x_range, y_range)
    ytic = [str(i + 1) for i in y_range]
    xtic = [str(i + 1) if i % 3 == 2 else '' for i in x_range]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    plt.xticks(y_range, ytic)
    plt.yticks(x_range, xtic)
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('Value')
    ax.plot_surface(Y, X, v, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                    vmin=-1.0, vmax=1.0, edgecolor='none')
    #plt.savefig(str(iter) + '.png')
    plt.show()


def plot_heatmap(array, title="", ax=None,
                 alpha=0.8, annot=True, fmt=".2f", cbar=True):
    """
    function used to generate heat maps
    """
    if ax is None:
        fig, ax = plt.subplot()
    sns.heatmap(array, ax=ax, cmap=plt.cm.rainbow, alpha=alpha, annot=annot, fmt=fmt, cbar=cbar)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xticklabels(range(1, 22))
    ax.set_yticklabels(range(1, 11))
    ax.set_xlabel("Player's sum")
    ax.set_ylabel("Dealer's sum")
    ax.set_title(title)
    return ax
