from src.eval.bacor_score import bacor_score
from src.eval.mjww_score import mjww_score
from src.eval.wufpc_score import wufpc_score
from src.eval.wtmf_score import wtmf_score
from src.eval.naive_bayes_score import nb_score
from src.eval.vocab_levenshtein_score import vocab_levenshtein_score
from src.eval.feature_extractions import show_topsegments_densities
from src.utils.plotters import plot_segment_mode_frequencies, plot_umm_confusion_matries
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from src.utils.eval_helpers import list2string, get_bacor_model
from src.eval.segment_statistics import get_average_segment_length, get_vocabulary_size, show_mode_segment_statistics

def single_iteration_pipeline(train_segmentation, train_modes, dev_segmentation, dev_modes):
    X_train, y_train = train_segmentation, train_modes
    X_test, y_test = dev_segmentation, dev_modes

    # bacor model
    train_data, test_data = list2string(X_train), list2string(X_test)
    bacor_model = get_bacor_model()
    bacor_model.fit(train_data, y_train)
    train_predictions = bacor_model.predict(train_data)
    dev_predictions = bacor_model.predict(test_data)
    # Bacor Accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    dev_accuracy = accuracy_score(y_test, dev_predictions)
    # Bacor F1
    train_f1 = f1_score(y_train, train_predictions, average='weighted')
    dev_f1 = f1_score(y_test, dev_predictions, average='weighted')
    # Melody Justified With Words score
    train_mjww, _, _ = mjww_score(train_segmentation)
    dev_mjww, _, _ = mjww_score(dev_segmentation, len(train_segmentation))
    # Weighted Top Mode Frequency score
    train_wtmf = wtmf_score(train_segmentation, train_modes)
    dev_wtmf = wtmf_score(dev_segmentation, dev_modes)
    # Weighted Unique Final Pitch Count score
    train_wufpc = wufpc_score(train_segmentation)
    dev_wufpc = wufpc_score(dev_segmentation)
    # Vocabulary Size
    train_vocab_size = get_vocabulary_size(train_segmentation)
    dev_vocab_size = get_vocabulary_size(dev_segmentation)
    # Average Segment Length
    train_avg_segment_len = get_average_segment_length(train_segmentation)
    dev_avg_segment_len = get_average_segment_length(dev_segmentation)
    # Average Segment Length
    train_vocab_levenhstein = vocab_levenshtein_score(train_segmentation)
    dev_vocab_levenhstein = vocab_levenshtein_score(dev_segmentation)

    return train_accuracy, train_f1, train_mjww, train_wtmf, train_wufpc, train_vocab_size, train_avg_segment_len, train_vocab_levenhstein,\
        dev_accuracy, dev_f1, dev_mjww, dev_wtmf, dev_wufpc, dev_vocab_size, dev_avg_segment_len, dev_vocab_levenhstein


def evaluation_pipeline(X_train, y_train, X_test, y_test, train_perplexity=-1, test_perplexity=-1, mjwp_score=-1,
        max_features_from_model = 100, max_features_additative = 100, include_additative = True, fe_occurence_coef=1):
    """
    final segmentation is a list of list of segments
    """
    # Compute NB scores
    train_nb_accuracy, train_nb_f1, test_nb_accuracy, test_nb_f1 = \
        nb_score(X_train, X_test, y_train, y_test)


    # Train dataset
    bacor, selected_features, trained_model = bacor_score(
        X_train, y_train, X_train, y_train,
        max_features_from_model = max_features_from_model,
        max_features_additative = max_features_additative,
        include_additative = include_additative,
        fe_occurence_coef = fe_occurence_coef
    )
    # Melody Justified With Words score
    mjww_words, mjww_segments, mjww_average = mjww_score(X_train)
    # Weighted Top Mode Frequency score
    wtmf = wtmf_score(X_train, y_train)
    # Weighted Unique Final Pitch Count score
    wufpc = wufpc_score(X_train)
    # Vocabulary Size
    vocab_size = get_vocabulary_size(X_train)
    # Average Segment Length
    avg_segment_len = get_average_segment_length(X_train)
    # Vocab Levenshtein Score
    vocab_levenshtein = vocab_levenshtein_score(X_train)


    # Print scores
    print("------------------------------- Train Scores -------------------------------")
    print()
    print("\t\t bacor accuracy and f1")
    print("\t\t\t accuracy: {:.2f}%".format(bacor["test"]["accuracy"]*100))
    print("\t\t\t f1: {:.2f}%".format(bacor["test"]["f1"]*100))
    print()
    print("\t\t NB accuracy and f1")
    print("\t\t\t accuracy: {:.2f}%".format(train_nb_accuracy*100))
    print("\t\t\t f1: {:.2f}%".format(train_nb_f1*100))
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
    print("\t\t\t words justification: {:.2f}% of segments".format(mjww_words*100))
    print("\t\t\t segments justification: {:.2f}% of segments".format(mjww_segments*100))
    print("\t\t\t average justification: {:.2f}% of segments".format(mjww_average*100))
    print()
    print("\t\t Weighted Top Mode Frequency")
    print("\t\t\t wtmf: {:.2f}% of melodies".format(wtmf*100))
    print()
    print("\t\t Weighted Unique Final Pitch Count")
    print("\t\t\t wufpc: {:.2f} final pitches for a chant".format(wufpc))
    print()
    print("\t\t Vocabulary Levenhstein Score")
    print("\t\t\t wufpc: {:.2f} final pitches for a chant".format(vocab_levenshtein))
    show_mode_segment_statistics(X_train, y_train)
    print("--------------------------------------------------------------------------")

    # Test dataset
    # bacor score
    bacor, selected_features, trained_model = bacor_score(
        X_train, y_train, X_test, y_test,
        max_features_from_model = max_features_from_model,
        max_features_additative = max_features_additative,
        include_additative = include_additative,
        fe_occurence_coef = fe_occurence_coef
    )

    # Melody Justified With Words score
    mjww_words, mjww_segments, mjww_average = mjww_score(X_test, len(X_train))
    # Weighted Top Mode Frequency score
    wtmf = wtmf_score(X_test, y_test)
    # Weighted Unique Final Pitch Count score
    wufpc = wufpc_score(X_test)
    # Vocabulary Size
    vocab_size = get_vocabulary_size(X_test)
    # Average Segment Length
    avg_segment_len = get_average_segment_length(X_test)
    # Vocab Levenshtein Score
    vocab_levenshtein = vocab_levenshtein_score(X_test)


    # Print scores
    print("------------------------------- Test Scores -------------------------------")
    print()
    print("\t\t bacor accuracy and f1")
    print("\t\t\t accuracy: {:.2f}%".format(bacor["test"]["accuracy"]*100))
    print("\t\t\t f1: {:.2f}%".format(bacor["test"]["f1"]*100))
    print()
    print("\t\t NB accuracy and f1")
    print("\t\t\t accuracy: {:.2f}%".format(test_nb_accuracy*100))
    print("\t\t\t f1: {:.2f}%".format(test_nb_f1*100))
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
    print("\t\t\t words justification: {:.2f}% of segments".format(mjww_words*100))
    print("\t\t\t segments justification: {:.2f}% of segments".format(mjww_segments*100))
    print("\t\t\t average justification: {:.2f}% of segments".format(mjww_average*100))
    print()
    print("\t\t Weighted Top Mode Frequency")
    print("\t\t\t wtmf: {:.2f}% of melodies".format(wtmf*100))
    print()
    print("\t\t Weighted Unique Final Pitch Count")
    print("\t\t\t wufpc: {:.2f} final pitches for a chant".format(wufpc))
    print()
    print("\t\t Vocabulary Levenhstein Score")
    print("\t\t\t wufpc: {:.2f} final pitches for a chant".format(vocab_levenshtein))
    show_mode_segment_statistics(X_test, y_test)
    print("--------------------------------------------------------------------------")

    print("Top selected melodies - from model: {}".format(selected_features["from_model"]["top_melodies"]))
    plot_segment_mode_frequencies(selected_features["from_model"]["melody_mode_frequencies"])
    show_topsegments_densities(X_train+X_test, y_train+y_test, set(selected_features["from_model"]["top_melodies"]))
    if include_additative:
        print("Top selected melodies - additative approach: {}".format(selected_features["additative"]["top_melodies"]))
        plot_segment_mode_frequencies(selected_features["additative"]["melody_mode_frequencies"])
        show_topsegments_densities(X_train+X_test, y_train+y_test, set(selected_features["additative"]["top_melodies"]))
    print()
    print()
    print("------------------------- Train + Test data charts ----------------------------")
    show_mode_segment_statistics(X_train+X_test, y_train + y_test)
    print("------------------------------------------------------------------------")

    return trained_model


def bacor_pipeline(final_segmentation, modes, train_len: int = 9706, test_len: int = 4159,
        max_features_from_model = 100, max_features_additative = 100, include_additative = True,
        fe_occurence_coef = 1):
    X_train, y_train = final_segmentation[:train_len], modes[:train_len]
    X_test, y_test = final_segmentation[train_len:], modes[train_len:]
    assert len(X_test) == test_len and len(y_test) == test_len

    # bacor score
    scores, selected_features, trained_model = bacor_score(
        X_train, y_train, X_test, y_test,
        max_features_from_model = max_features_from_model,
        max_features_additative = max_features_additative,
        include_additative = include_additative,
        fe_occurence_coef = fe_occurence_coef
    )

    # print scores
    print("Top selected melodies - from model: {}".format(selected_features["from_model"]["top_melodies"]))
    plot_segment_mode_frequencies(selected_features["from_model"]["melody_mode_frequencies"])
    if include_additative:
        print("Top selected melodies - additative approach: {}".format(selected_features["additative"]["top_melodies"]))
        plot_segment_mode_frequencies(selected_features["additative"]["melody_mode_frequencies"])

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

    train_predictions, _ = umm_model.predict_modes(train_chants,
                            final_range_classifier = final_range_classifier,
                            mode_priors_uniform = mode_priors_uniform)
    dev_predictions, _ = umm_model.predict_modes(dev_chants,
                            final_range_classifier = final_range_classifier,
                            mode_priors_uniform = mode_priors_uniform)
    test_predictions, _ = umm_model.predict_modes(test_chants,
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
        max_features_from_model = 100, max_features_additative = 100, include_additative = False, left_trim = 1, right_trim = 1):
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
        if len(chant) >= 1 + left_trim + right_trim:
            if right_trim == 0:
                trimmed_X_train.append(chant[left_trim:])
            else:
                trimmed_X_train.append(chant[left_trim:-right_trim])
            trimmed_y_train.append(mode)
        else:
            not_trimmed_train += 1
    # Trim test dataset
    for chant, mode in zip(X_test, y_test):
        if len(chant) >= 1 + left_trim + right_trim:
            if right_trim == 0:
                trimmed_X_test.append(chant[left_trim:])
            else:
                trimmed_X_test.append(chant[left_trim:-right_trim])
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