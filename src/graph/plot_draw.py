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


def plot_combined_recall(lstm, gru, nn, cnn, filename):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(lstm.history['recall'])
    ax.plot(gru.history['recall'])
    ax.plot(nn.history['recall'])
    ax.plot(cnn.history['recall'])
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend(["RNN with LSTM", "RNN with GRU", "Simple NN", "CNN"])
    plt.savefig(plot_path + filename)
    plt.show()


def plot_combined_precision(lstm, gru, nn, cnn, filename):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(lstm.history['val_precision'])
    ax.plot(gru.history['val_precision'])
    ax.plot(nn.history['val_precision'])
    ax.plot(cnn.history['val_precision'])
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.legend(["RNN with LSTM", "RNN with GRU", "Simple NN", "CNN"])
    plt.savefig(plot_path + filename)
    plt.show()