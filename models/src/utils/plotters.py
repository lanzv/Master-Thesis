import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_mode_segment_statistics(
        shared_dataframe: DataFrame,
        distinct_dataframe: DataFrame,
        vocabulary_sizes_dataframe: DataFrame,
        unique_dataframe: DataFrame):
    """
    Plot statisics of modes and unique segments of segmented chants.
    There are 4 confusion matrices, two 8x8, two 8x1.
    First chart - shared segments over modes (the upper triangular matrix is all we need - the rest is a copy)
    Second chart - distinct segments over modes (-||-)
    Third chart - vocabulary size of each mode
    Fourth chart - unique vocabulary of the specific mode

    Parameters
    ----------
    shared_dataframe : DataFrame
        dataframe of columns and index, both [1..8], and table 8x8 with computed shared segments
    distinct_dataframe : DataFrame
        dataframe of columns and index, both [1..8], and table 8x8 with computed distinct segments
    vocabulary_sizes_dataframe : DataFrame
        dataframe of columns and index, [1..8], and 8x1 table with computed vocabulary sizes
    unique_dataframe : DataFrame
        dataframe of columns and index, [1..8], and 8x1 table with computed vocabulary sizes
    """
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
    """
    Plot 8 densities charts (both x,y scales are measured in percentes) of segment uniqueness.
    Each chart is mapped to one mode. The chart shows us the statistics about unique segments, whether
    there are statistically more unique segments at beggining or end or in the middle of chants.

    Parameters
    ----------
    densities : dictionary
        dictionary of modes of numpy arrays - size of each array is 400, which corresponds to 100%
        (index 0 corresponds to 0%, index 3 corresponds to 1% ... ), value at each index contains the
        percentage ratio of num_unique_segments/num_all_segments at the specific part (% of the whole chant)
        of chant.
    labels : list
        list of modes
    """
    figure, axis = plt.subplots(1, len(labels))
    for i, label in enumerate(labels):
        axis[i].plot(np.arange(0.25, 100.25, 0.25), densities[label])
        axis[i].set_title("Density of unique segments for mode {}".format(label))
        axis[i].set_xlabel("chant position (%)")
        axis[i].set_ylabel("unique occurences (%)")
    figure.set_size_inches(40, 5)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.2, hspace=0.2)
    plt.show()



def plot_topsegments_densities(densities: dict, labels = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    Plot 8 densities charts (both x,y scales are measured in percentes) of top segment picked from
    Feature Extraction. Each chart is mapped to one mode. The chart shows us the statistics about 
    top segments, whether there are statistically more important segments at beggining or end or 
    in the middle of chants.

    Parameters
    ----------
    densities : dictionary
        dictionary of modes of numpy arrays - size of each array is 400, which corresponds to 100%
        (index 0 corresponds to 0%, index 3 corresponds to 1% ... ), value at each index contains the
        percentage ratio of num_top_segments/num_all_segments at the specific part (% of the whole chant)
        of chant.
    labels : list
        list of modes
    """
    figure, axis = plt.subplots(1, len(labels))
    for i, label in enumerate(labels):
        axis[i].plot(np.arange(0.25, 100.25, 0.25), densities[label])
        axis[i].set_title("Density of top segments for mode {}".format(label))
        axis[i].set_xlabel("chant position (%)")
        axis[i].set_ylabel("unique occurences (%)")
    figure.set_size_inches(40, 5)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.2, hspace=0.2)
    plt.show()





def plot_segment_mode_frequencies(frequencies: DataFrame):
    """
    Plot confusion matrix of segments get from feature extraction.
    The matrix will show the information about segment occurences over all modes of selected segments.
    All columns should be summed up to 1 in the comming frequencies argument.

    Parameters
    ----------
    frequencies : DataFrame
        DataFrame of index [1, ..., 8], columns ["asasd 2500", ...] (first value is segment, second value is
        number of occurences over all segments), data (frequencies of mode occurences of each selected segment
        that the sum of them should give 1)
    """
    f = plt.figure()
    f.set_figwidth(len(frequencies.columns)*2)
    f.set_figheight(len(frequencies.index)*2)
    plt.pcolor(frequencies)
    plt.yticks(np.arange(0.5, len(frequencies.index), 1), frequencies.index)
    plt.xticks(np.arange(0.5, len(frequencies.columns), 1), frequencies.columns)
    plt.show()




def plot_iteration_statistics(statistics_to_plot):
    """
    Plot charts for all scores that are computed during model training over specified iterations.

    Parameters
    ----------
    statistics_to_plot : dictionary
        dictionary where key is a chart label, value is a tuple
        of two arrays - 1. iterations we are printing 2. score progress, ([5, 10, 15], [0.3, 0.4, 0.5])
    """
    figure, axis = plt.subplots(1, len(statistics_to_plot))
    for i, title in enumerate(statistics_to_plot):
        x, y = statistics_to_plot[title]
        axis[i].plot(x, y)
        axis[i].set_title(title)
    figure.set_size_inches(40, 5)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.2, hspace=0.2)
    plt.show()




def plot_umm_confusion_matries(train_true, train_pred, dev_true, dev_pred, test_true, test_pred, labels):
    """
    Plot confusion matrices of train, dev and test datasets of predicted/true modes.
    The function is implemented mainly for the UMM mode prediction
    (based on Bayessian rule or other classifier).

    Parameters
    ----------
    train_true : list of chars
        list of true modes of training data
    train_pred : list of chars
        list of predicted modes of training data
    dev_true : list of chars
        list of true modes of dev data
    dev_pred : list of chars
        list of predicted modes of dev data
    test_true : list of chars
        list of true modes of testing data
    test_pred : list of chars
        list of predicted modes of testing data
    labels : list of chars
        list of modes [1, 2, 3, 4, .., 8]
    """
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



def plot_trimmed_segments(trimmed_scores):
    """
    Plot charts of trimmed segments. Visualize comparison of trimming from left, right or both sides with the random baseline.
    First, plot accuracy.
    Second, plot F1.

    Parameters
    ----------
    trimmed_scores : dict
        dict that contains nine keys: 'trimmed segments', 'left accuracy', 'right accuracy, 'both sides accuracy',
        'left f1', 'right f1, 'both sides f1', 'random segments accuracy', 'random segments f1' each their values 
        are lists of scores, only trimmed segments is a list of number of trimmed segments
    """
    # Plot accuracy
    # create data
    x = trimmed_scores["trimmed segments"]
    y_left = trimmed_scores["left accuracy"]
    y_right = trimmed_scores["right accuracy"]
    y_both = trimmed_scores["both sides accuracy"]
    y_rand = trimmed_scores["random segments accuracy"]
    
    # plot lines
    plt.plot(x, y_left, label = "left trimmed")
    plt.plot(x, y_right, label = "right trimmed")
    plt.plot(x, y_both, label = "both side trimmed")
    plt.plot(x, y_rand, label = "baseline - random trimmed", color='gray', linestyle='dashed')
    plt.xticks(x)
    plt.legend()
    plt.show()

    # Plot F1
    # create data
    x = trimmed_scores["trimmed segments"]
    y_left = trimmed_scores["left f1"]
    y_right = trimmed_scores["right f1"]
    y_both = trimmed_scores["both sides f1"]
    y_rand = trimmed_scores["random segments f1"]
    
    # plot lines
    plt.plot(x, y_left, label = "left trimmed")
    plt.plot(x, y_right, label = "right trimmed")
    plt.plot(x, y_both, label = "both side trimmed")
    plt.plot(x, y_rand, label = "baseline - random trimmed", color='gray', linestyle='dashed')
    plt.xticks(x)
    plt.legend()
    plt.show()