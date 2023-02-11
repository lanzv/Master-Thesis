from src.eval.bacor_score import bacor_score
from src.eval.mjww_score import mjww_score
from src.eval.wufpc_score import wufpc_score
from src.eval.wtmf_score import wtmf_score
from src.utils.plotters import plot_melody_mode_frequencies

def evaluation_pipeline(final_segmentation, modes, train_len: int = 9706, test_len: int = 4159, max_features = 100):
    X_train, y_train = final_segmentation[:train_len], modes[:train_len]
    X_test, y_test = final_segmentation[train_len:], modes[train_len:]
    assert len(X_test) == test_len and len(y_test) == test_len

    # bacor score
    bacor, selected_features, trained_model = bacor_score(
        X_train, y_train, X_test, y_test, max_features = max_features
    )
    # Melody Justified With Words score
    mjww = mjww_score(final_segmentation)
    # Weighted Top Mode Frequency score
    wtmf = wtmf_score(final_segmentation, modes)
    # Weighted Unique Final Pitch Count score
    wufpc = wufpc_score(final_segmentation)


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
    print("--------------------------------------------------------------------------")

    print("Top selected melodies - from model: {}".format(selected_features["from_model"]["top_melodies"]))
    plot_melody_mode_frequencies(selected_features["from_model"]["melody_mode_frequencies"])
    print("Top selected melodies - additative approach: {}".format(selected_features["additative"]["top_melodies"]))
    plot_melody_mode_frequencies(selected_features["additative"]["melody_mode_frequencies"])


    return bacor, mjww, wtmf, wufpc, selected_features, trained_model


def bacor_pipeline(final_segmentation, modes, train_len: int = 9706, test_len: int = 4159,
        max_features_from_model = 100, max_features_additative = 100):
    X_train, y_train = final_segmentation[:train_len], modes[:train_len]
    X_test, y_test = final_segmentation[train_len:], modes[train_len:]
    assert len(X_test) == test_len and len(y_test) == test_len

    # bacor score
    scores, selected_features, trained_model = bacor_score(
        X_train, y_train, X_test, y_test,
        max_features_from_model = max_features_from_model,
        max_features_additative = max_features_additative
    )

    # print scores
    print("Top selected melodies - from model: {}".format(selected_features["from_model"]["top_melodies"]))
    plot_melody_mode_frequencies(selected_features["from_model"]["melody_mode_frequencies"])
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