import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




def plot_mode_segment_statistics(
        shared_dataframe: DataFrame,
        distinct_dataframe: DataFrame,
        vocabulary_sizes_dataframe: DataFrame,
        unique_dataframe: DataFrame):
    
    figure, axis = plt.subplots(1, 4)

    figure.set_size_inches(25, 5)
    # Same segments in mode pair
    axis[0].matshow(shared_dataframe, interpolation ='nearest')
    axis[0].set_xticks(np.arange(len(shared_dataframe.index)), minor=False)
    axis[0].set_xticklabels(shared_dataframe.index, fontdict=None, minor=False)
    axis[0].set_yticks(np.arange(len(shared_dataframe.columns)), minor=False)
    axis[0].set_yticklabels(shared_dataframe.columns, fontdict=None, minor=False)
    axis[0].xaxis.set_ticks_position('bottom')
    for (i, j), z in np.ndenumerate(shared_dataframe):
        axis[0].text(j, i, '{}'.format(int(z)), ha='center', va='center')
    axis[0].set_title("Shared Segments")


    # Different segments in mode pair
    axis[1].matshow(distinct_dataframe, interpolation ='nearest')
    axis[1].set_xticks(np.arange(len(distinct_dataframe.index)), minor=False)
    axis[1].set_xticklabels(distinct_dataframe.index, fontdict=None, minor=False)
    axis[1].set_yticks(np.arange(len(distinct_dataframe.columns)), minor=False)
    axis[1].set_yticklabels(distinct_dataframe.columns, fontdict=None, minor=False)
    axis[1].xaxis.set_ticks_position('bottom')
    for (i, j), z in np.ndenumerate(distinct_dataframe):
        axis[1].text(j, i, '{}'.format(int(z)), ha='center', va='center')
    axis[1].set_title("Distinct Segments")


    # Vocabulary Size in modes
    axis[2].matshow(vocabulary_sizes_dataframe, interpolation ='nearest')
    axis[2].set_yticks(np.arange(len(vocabulary_sizes_dataframe.index)), minor=False)
    axis[2].set_yticklabels(vocabulary_sizes_dataframe.index, fontdict=None, minor=False)
    axis[2].get_xaxis().set_visible(False)
    for (i, j), z in np.ndenumerate(vocabulary_sizes_dataframe):
        axis[2].text(j, i, '{}'.format(int(z)), ha='center', va='center')
    axis[2].set_title("Vocabulary Size")

    # Unique Segments in modes
    axis[3].matshow(unique_dataframe, interpolation ='nearest')
    axis[3].set_yticks(np.arange(len(unique_dataframe.index)), minor=False)
    axis[3].set_yticklabels(unique_dataframe.index, fontdict=None, minor=False)
    axis[3].get_xaxis().set_visible(False)
    for (i, j), z in np.ndenumerate(unique_dataframe):
        axis[3].text(j, i, '{}'.format(int(z)), ha='center', va='center')
    axis[3].set_title("Unique Segments")

    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.2, hspace=0.2)
    plt.show()


def plot_unique_segments_densities(densities: dict, labels = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    figure, axis = plt.subplots(1, len(labels))
    for i, label in enumerate(labels):
        axis[i].plot(np.arange(0.25, 100.25, 0.25), densities[label])
        axis[i].set_title("Density of unique segments for mode {}".format(label))
        axis[i].set_xlabel("chant position (%)")
        axis[i].set_ylabel("unique occurences (%)")
    figure.set_size_inches(40, 5)
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

def plot_iteration_statistics(statistics_to_plot):
    figure, axis = plt.subplots(1, len(statistics_to_plot))
    for i, title in enumerate(statistics_to_plot):
        x, y = statistics_to_plot[title]
        axis[i].plot(x, y)
        axis[i].set_title(title)
    figure.set_size_inches(40, 5)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.2, hspace=0.2)
    plt.show()

def plot_umm_confusion_matries(train_true, train_pred, dev_true, dev_pred, test_true, test_pred, labels):
    print("Train UMM modes accuracy")
    cm = confusion_matrix(train_true, train_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot()
    plt.show()
    print("Dev UMM modes accuracy")
    cm = confusion_matrix(dev_true, dev_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot()
    plt.show()
    print("Test UMM modes accuracy")
    cm = confusion_matrix(test_true, test_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot()
    plt.show()