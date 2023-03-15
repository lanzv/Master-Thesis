import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


def plot_mode_segment_statistics(
        shared_dataframe: DataFrame,
        distinct_dataframe: DataFrame,
        vocabulary_sizes_dataframe: DataFrame,
        unique_dataframe: DataFrame):
    
    figure, axis = plt.subplots(1, 4)

    # Same segments in mode pair
    axis[0].matshow(shared_dataframe, interpolation ='nearest')
    axis[0].set_xticks(np.arange(len(shared_dataframe.index)), minor=False)
    axis[0].set_xticklabels(shared_dataframe.index, fontdict=None, minor=False)
    axis[0].set_yticks(np.arange(len(shared_dataframe.columns)), minor=False)
    axis[0].set_yticklabels(shared_dataframe.columns, fontdict=None, minor=False)
    axis[0].xaxis.set_ticks_position('bottom')
    for (i, j), z in np.ndenumerate(shared_dataframe):
        axis[0].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    axis[0].set_title("Shared Segments")


    # Different segments in mode pair
    axis[1].matshow(distinct_dataframe, interpolation ='nearest')
    axis[1].set_xticks(np.arange(len(distinct_dataframe.index)), minor=False)
    axis[1].set_xticklabels(distinct_dataframe.index, fontdict=None, minor=False)
    axis[1].set_yticks(np.arange(len(distinct_dataframe.columns)), minor=False)
    axis[1].set_yticklabels(distinct_dataframe.columns, fontdict=None, minor=False)
    axis[1].xaxis.set_ticks_position('bottom')
    for (i, j), z in np.ndenumerate(distinct_dataframe):
        axis[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    axis[1].set_title("Distinct Segments")


    # Vocabulary Size in modes
    axis[2].matshow(vocabulary_sizes_dataframe, interpolation ='nearest')
    axis[2].set_xticks(np.arange(len(vocabulary_sizes_dataframe.index)), minor=False)
    axis[2].set_xticklabels(vocabulary_sizes_dataframe.index, fontdict=None, minor=False)
    axis[2].set_yticks(np.arange(len(vocabulary_sizes_dataframe.columns)), minor=False)
    axis[2].set_yticklabels(vocabulary_sizes_dataframe.columns, fontdict=None, minor=False)
    axis[2].xaxis.set_ticks_position('bottom')
    for (i, j), z in np.ndenumerate(vocabulary_sizes_dataframe):
        axis[2].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    axis[2].set_title("Vocabulary Size")

    # Unique Segments in modes
    axis[3].matshow(unique_dataframe, interpolation ='nearest')
    axis[3].set_xticks(np.arange(len(unique_dataframe.index)), minor=False)
    axis[3].set_xticklabels(unique_dataframe.index, fontdict=None, minor=False)
    axis[3].set_yticks(np.arange(len(unique_dataframe.columns)), minor=False)
    axis[3].set_yticklabels(unique_dataframe.columns, fontdict=None, minor=False)
    axis[3].xaxis.set_ticks_position('bottom')
    for (i, j), z in np.ndenumerate(unique_dataframe):
        axis[3].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    axis[3].set_title("Unique Segments")

    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.2, hspace=0.2)
    plt.show()


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