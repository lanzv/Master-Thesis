import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


def plot_melody_mode_frequencies(frequencies: DataFrame):
    f = plt.figure()
    f.set_figwidth(len(frequencies.columns)*2)
    f.set_figheight(len(frequencies.index)*2)
    plt.pcolor(frequencies)
    plt.yticks(np.arange(0.5, len(frequencies.index), 1), frequencies.index)
    plt.xticks(np.arange(0.5, len(frequencies.columns), 1), frequencies.columns)
    plt.show()


def plot_line_chart(title, x, y):
    plt.plot(x, y)
    plt.title(title)
    plt.show()