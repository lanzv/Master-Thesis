from src.eval.bacor_score import bacor_score
from src.eval.mjww_score import mjww_score
from src.eval.wufpc_score import wufpc_score
from src.eval.wtmf_score import wtmf_score
from src.utils.plotters import plot_melody_mode_frequencies
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from src.utils.eval_helpers import list2string, get_bacor_model
from src.eval.segment_statistics import get_average_segment_length, get_vocabulary_size, show_mode_segment_statistics
import logging

def single_iteration_pipeline(final_segmentation, modes, iteration, top_melodies):
    if len(final_segmentation) == 9706 + 4159:
        train_len = 9706
    else:
        logging.info("Iteration Pipeline: different length of training/testing datasets: {} chants."
            .format(len(final_segmentation)))
        train_len = int(0.75*len(final_segmentation))

    X_train, y_train = final_segmentation[:train_len], modes[:train_len]
    X_test, y_test = final_segmentation[train_len:], modes[train_len:]

    # bacor score
    train_data, test_data = list2string(X_train), list2string(X_test)
    bacor_model = get_bacor_model()
    bacor_model.fit(train_data, y_train)
    predictions = bacor_model.predict(test_data)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    # Melody Justified With Words score
    mjww = mjww_score(final_segmentation)
    # Weighted Top Mode Frequency score
    wtmf = wtmf_score(final_segmentation, modes)
    # Weighted Unique Final Pitch Count score
    wufpc = wufpc_score(final_segmentation)
    # Vocabulary Size
    vocab_size = get_vocabulary_size(final_segmentation)
    # Average Segment Length
    avg_segment_len = get_average_segment_length(final_segmentation)

    print("{}. Iteration \t accuracy: {:.2f}%, f1: {:.2f}%, mjww: {:.2f}%, wtmf: {:.2f}%, wufpc: {:.2f} pitches, vocabulary size: {}, avg segment len: {:.2f} \t\t {}"
        .format(iteration, accuracy*100, f1*100, mjww*100, wtmf*100, wufpc, vocab_size, avg_segment_len, top_melodies))
    return accuracy, f1, mjww, wtmf, wufpc, vocab_size, avg_segment_len


def evaluation_pipeline(final_segmentation, modes, train_len: int = 9706, test_len: int = 4159,
        max_features_from_model = 100, max_features_additative = 100, include_additative = True):
    """
    final segmentation is a list of list of segments
    """
    X_train, y_train = final_segmentation[:train_len], modes[:train_len]
    X_test, y_test = final_segmentation[train_len:], modes[train_len:]
    assert len(X_test) == test_len and len(y_test) == test_len

    # bacor score
    bacor, selected_features, trained_model = bacor_score(
        X_train, y_train, X_test, y_test,
        max_features_from_model = max_features_from_model,
        max_features_additative = max_features_additative,
        include_additative = include_additative
    )
    # Melody Justified With Words score
    mjww = mjww_score(final_segmentation)
    # Weighted Top Mode Frequency score
    wtmf = wtmf_score(final_segmentation, modes)
    # Weighted Unique Final Pitch Count score
    wufpc = wufpc_score(final_segmentation)
    # Vocabulary Size
    vocab_size = get_vocabulary_size(final_segmentation)
    # Average Segment Length
    avg_segment_len = get_average_segment_length(final_segmentation)


    # Print scores
    print("--------------------------------- Scores ---------------------------------")
    print()
    print("\t\t bacor accuracy and f1")
    print("\t\t\t accuracy: {:.2f}%".format(bacor["test"]["accuracy"]*100))
    print("\t\t\t f1: {:.2f}%".format(bacor["test"]["f1"]*100))
    print()
    print("\t\t Melody Justified With Words")
    print("\t\t\t mjww: {:.2f}% of segments".format(mjww*100))
    print()
    print("\t\t Weighted Top Mode Frequency")
    print("\t\t\t wtmf: {:.2f}% of melodies".format(wtmf*100))
    print()
    print("\t\t Weighted Unique Final Pitch Count")
    print("\t\t\t wufpc: {:.2f} final pitches for a chant".format(wufpc))
    print()
    print("\t\t Vocabulary Size")
    print("\t\t\t size: {} unique segments".format(vocab_size))
    print()
    print("\t\t Average Segment Length")
    print("\t\t\t avgerage: {:.2f} tones in one segment".format(avg_segment_len))
    print("--------------------------------------------------------------------------")

    print("Top selected melodies - from model: {}".format(selected_features["from_model"]["top_melodies"]))
    plot_melody_mode_frequencies(selected_features["from_model"]["melody_mode_frequencies"])
    if include_additative:
        print("Top selected melodies - additative approach: {}".format(selected_features["additative"]["top_melodies"]))
        plot_melody_mode_frequencies(selected_features["additative"]["melody_mode_frequencies"])

    show_mode_segment_statistics(final_segmentation, modes)

    return bacor, mjww, wtmf, wufpc, selected_features, trained_model


def bacor_pipeline(final_segmentation, modes, train_len: int = 9706, test_len: int = 4159,
        max_features_from_model = 100, max_features_additative = 100, include_additative = True):
    X_train, y_train = final_segmentation[:train_len], modes[:train_len]
    X_test, y_test = final_segmentation[train_len:], modes[train_len:]
    assert len(X_test) == test_len and len(y_test) == test_len

    # bacor score
    scores, selected_features, trained_model = bacor_score(
        X_train, y_train, X_test, y_test,
        max_features_from_model = max_features_from_model,
        max_features_additative = max_features_additative,
        include_additative = include_additative
    )

    # print scores
    print("Top selected melodies - from model: {}".format(selected_features["from_model"]["top_melodies"]))
    plot_melody_mode_frequencies(selected_features["from_model"]["melody_mode_frequencies"])
    if include_additative:
        print("Top selected melodies - additative approach: {}".format(selected_features["additative"]["top_melodies"]))
        plot_melody_mode_frequencies(selected_features["additative"]["melody_mode_frequencies"])

    print("Train scores \n\t Precision: {:.2f}% \n\t Recall: {:.2f}% \n\t F1: {:.2f}% \n\t Accuracy: {:.2f}%"
        .format(scores["train"]["precision"]*100,
              scores["train"]["recall"]*100,
              scores["train"]["f1"]*100,
              scores["train"]["accuracy"]*100))
    print("Test scores \n\t Precision: {:.2f}% \n\t Recall: {:.2f}% \n\t F1: {:.2f}% \n\t Accuracy: {:.2f}%"
        .format(scores["test"]["precision"]*100,
                scores["test"]["recall"]*100,
                scores["test"]["f1"]*100,
                scores["test"]["accuracy"]*100))


    return scores, selected_features, trained_model