import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import os

plot_path = str(Path(__file__).parents[2]) + os.path.sep + 'plot' + os.path.sep


def plot_graphs(history, string,title, filename):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(history.history[string])
    ax.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend(["Training " + string, 'Test '+string])
    plt.title(title, fontsize=20)
    plt.savefig(plot_path + filename)
    plt.show()