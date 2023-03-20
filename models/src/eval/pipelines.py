from src.eval.bacor_score import bacor_score
from src.eval.mjww_score import mjww_score
from src.eval.wufpc_score import wufpc_score
from src.eval.wtmf_score import wtmf_score
from src.utils.plotters import plot_melody_mode_frequencies, plot_umm_confusion_matries
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from src.utils.eval_helpers import list2string, get_bacor_model
from src.eval.segment_statistics import get_average_segment_length, get_vocabulary_size, show_mode_segment_statistics

def single_iteration_pipeline(train_segmentation, train_modes, dev_segmentation, dev_modes, top_melodies):
    X_train, y_train = train_segmentation, train_modes
    X_test, y_test = dev_segmentation, dev_modes

    # bacor score
    train_data, test_data = list2string(X_train), list2string(X_test)
    bacor_model = get_bacor_model()
    bacor_model.fit(train_data, y_train)
    predictions = bacor_model.predict(test_data)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    # Melody Justified With Words score
    mjww = mjww_score(dev_segmentation)
    # Weighted Top Mode Frequency score
    wtmf = wtmf_score(dev_segmentation, dev_modes)
    # Weighted Unique Final Pitch Count score
    wufpc = wufpc_score(dev_segmentation)
    # Vocabulary Size
    vocab_size = get_vocabulary_size(dev_segmentation)
    # Average Segment Length
    avg_segment_len = get_average_segment_length(dev_segmentation)

    return accuracy, f1, mjww, wtmf, wufpc, vocab_size, avg_segment_len, top_melodies


def evaluation_pipeline(X_train, y_train, X_test, y_test, train_perplexity=-1, test_perplexity=-1, mjwp_score=-1,
        max_features_from_model = 100, max_features_additative = 100, include_additative = True):
    """
    final segmentation is a list of list of segments
    """
    # Train dataset
    bacor, selected_features, trained_model = bacor_score(
        X_train, y_train, X_train, y_train,
        max_features_from_model = max_features_from_model,
        max_features_additative = max_features_additative,
        include_additative = include_additative
    )

    # Melody Justified With Words score
    mjww = mjww_score(X_train)
    # Weighted Top Mode Frequency score
    wtmf = wtmf_score(X_train, y_train)
    # Weighted Unique Final Pitch Count score
    wufpc = wufpc_score(X_train)
    # Vocabulary Size
    vocab_size = get_vocabulary_size(X_train)
    # Average Segment Length
    avg_segment_len = get_average_segment_length(X_train)


    # Print scores
    print("------------------------------- Train Scores -------------------------------")
    print()
    print("\t\t bacor accuracy and f1")
    print("\t\t\t accuracy: {:.2f}%".format(bacor["test"]["accuracy"]*100))
    print("\t\t\t f1: {:.2f}%".format(bacor["test"]["f1"]*100))
    print()
    print("\t\t Perplexity")
    print("\t\t\t {:.6f}".format(train_perplexity))
    print()
    print("\t\t Vocabulary Size")
    print("\t\t\t size: {} unique segments".format(vocab_size))
    print()
    print("\t\t Average Segment Length")
    print("\t\t\t avgerage: {:.2f} tones in one segment".format(avg_segment_len))
    print()
    print("\t\t Melody Justified With Words")
    print("\t\t\t mjww: {:.2f}% of segments".format(mjww*100))
    print()
    print("\t\t Weighted Top Mode Frequency")
    print("\t\t\t wtmf: {:.2f}% of melodies".format(wtmf*100))
    print()
    print("\t\t Weighted Unique Final Pitch Count")
    print("\t\t\t wufpc: {:.2f} final pitches for a chant".format(wufpc))
    show_mode_segment_statistics(X_train, y_train)
    print("--------------------------------------------------------------------------")

    # Test dataset
    # bacor score
    bacor, selected_features, trained_model = bacor_score(
        X_train, y_train, X_test, y_test,
        max_features_from_model = max_features_from_model,
        max_features_additative = max_features_additative,
        include_additative = include_additative
    )

    # Melody Justified With Words score
    mjww = mjww_score(X_test)
    # Weighted Top Mode Frequency score
    wtmf = wtmf_score(X_test, y_test)
    # Weighted Unique Final Pitch Count score
    wufpc = wufpc_score(X_test)
    # Vocabulary Size
    vocab_size = get_vocabulary_size(X_test)
    # Average Segment Length
    avg_segment_len = get_average_segment_length(X_test)


    # Print scores
    print("------------------------------- Test Scores -------------------------------")
    print()
    print("\t\t bacor accuracy and f1")
    print("\t\t\t accuracy: {:.2f}%".format(bacor["test"]["accuracy"]*100))
    print("\t\t\t f1: {:.2f}%".format(bacor["test"]["f1"]*100))
    print()
    print("\t\t Perplexity")
    print("\t\t\t {:.6f}".format(test_perplexity))
    print()
    print("\t\t Vocabulary Size")
    print("\t\t\t size: {} unique segments".format(vocab_size))
    print()
    print("\t\t Average Segment Length")
    print("\t\t\t avgerage: {:.2f} tones in one segment".format(avg_segment_len))
    print()
    print("\t\t Melody Justified With Phrases")
    print("\t\t\t mjww: {:.2f}% of segments".format(mjwp_score*100))
    print()
    print("\t\t Melody Justified With Words")
    print("\t\t\t mjww: {:.2f}% of segments".format(mjww*100))
    print()
    print("\t\t Weighted Top Mode Frequency")
    print("\t\t\t wtmf: {:.2f}% of melodies".format(wtmf*100))
    print()
    print("\t\t Weighted Unique Final Pitch Count")
    print("\t\t\t wufpc: {:.2f} final pitches for a chant".format(wufpc))
    show_mode_segment_statistics(X_test, y_test)
    print("--------------------------------------------------------------------------")

    print("Top selected melodies - from model: {}".format(selected_features["from_model"]["top_melodies"]))
    plot_melody_mode_frequencies(selected_features["from_model"]["melody_mode_frequencies"])
    if include_additative:
        print("Top selected melodies - additative approach: {}".format(selected_features["additative"]["top_melodies"]))
        plot_melody_mode_frequencies(selected_features["additative"]["melody_mode_frequencies"])
    print()
    print()
    print("------------------------- Train + Test data charts ----------------------------")
    show_mode_segment_statistics(X_train+X_test, y_train + y_test)
    print("------------------------------------------------------------------------")

    return trained_model


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

def umm_modes_accuracy_pipeline(umm_model, train_chants, train_modes, test_chants, test_modes,
                                train_proportion: float = 0.9, final_range_classifier = False,
                                mode_priors_uniform = True, labels = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    # Divide chants to train and dev datasets
    splitting_point = int(train_proportion*len(train_chants))
    train_chants, dev_chants = train_chants[:splitting_point], train_chants[splitting_point:]
    train_modes, dev_modes = train_modes[:splitting_point], train_modes[splitting_point:]

    train_predictions = umm_model.predict_modes(train_chants,
                            final_range_classifier = final_range_classifier,
                            mode_priors_uniform = mode_priors_uniform)
    dev_predictions = umm_model.predict_modes(dev_chants,
                            final_range_classifier = final_range_classifier,
                            mode_priors_uniform = mode_priors_uniform)
    test_predictions = umm_model.predict_modes(test_chants,
                            final_range_classifier = final_range_classifier,
                            mode_priors_uniform = mode_priors_uniform)

    print("------------------------ Unigram Model Modes - Mode Prediction Scores ------------------------")
    print()
    print("\t\tTrain accuracy {:.2f}%".format(100*accuracy_score(train_modes, train_predictions)))
    print("\t\tDev accuracy {:.2f}%".format(100*accuracy_score(dev_modes, dev_predictions)))
    print("\t\tTest accuracy {:.2f}%".format(100*accuracy_score(test_modes, test_predictions)))
    print()
    plot_umm_confusion_matries(train_modes, train_predictions,
                               dev_modes, dev_predictions,
                               test_modes, test_predictions,
                               labels = labels)
    print("----------------------------------------------------------------------------------------------")


def evaluation_trimmed_chants(X_train, y_train, X_test, y_test, train_perplexity = -1, test_perplexity = -1, mjwp = -1,
        max_features_from_model = 100, max_features_additative = 100, include_additative = False):
    """
    Remove first and last segment (if possible) and call evaluation pipeline
    """
    trimmed_X_train = []
    trimmed_y_train = []
    not_trimmed_train = 0
    trimmed_X_test = []
    trimmed_y_test = []
    not_trimmed_test = 0
    # Trim train dataset
    for chant, mode in zip(X_train, y_train):
        if len(chant) >= 3:
            trimmed_X_train.append(chant[1:-1])
            trimmed_y_train.append(mode)
        else:
            not_trimmed_train += 1
    # Trim test dataset
    for chant, mode in zip(X_test, y_test):
        if len(chant) >= 3:
            trimmed_X_test.append(chant[1:-1])
            trimmed_y_test.append(mode)
        else:
            not_trimmed_test += 1
    print("------------------- Trimmed Evalution Pipeline -----------------")
    print()
    print("\t\t {} train chants are too short to not to be trimmed".format(not_trimmed_train))
    print("\t\t {} test chants are too short to not to be trimmed".format(not_trimmed_test))
    # call evaluation pipeline
    return evaluation_pipeline(trimmed_X_train, trimmed_y_train, trimmed_X_test, trimmed_y_test,
                               train_perplexity, test_perplexity, mjwp, max_features_from_model,
                               max_features_additative, include_additative)