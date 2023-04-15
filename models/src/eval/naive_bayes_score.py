from sklearn.metrics import accuracy_score, f1_score
from src.utils.eval_helpers import get_nb_model
from src.utils.eval_helpers import list2string



def nb_score(train_segments, test_segments, train_modes, test_modes):
    """
    Fit the Naive Bayes model by training segmentations in order to predit modes.
    Compute its accuracy and f1 for both, training and testing segmentations.

    Parameters
    ----------
    train_segments : list of list of strings
        list of trainig chant segmentation represented as string of segment list
        example: [["aas", "sss", "aaab"], ["bb", "kj", "l", "l", "ajjkjh"]]
    test_segments : list of list of strings
        list of testing chant segmentation represented as string of segment list
        example: [["aas", "sss", "aaab"], ["bb", "kj", "l", "l", "ajjkjh"]]
    train_modes : list of strings
        list of modes of training chants
    test_modes : list of strings
        list of modes of testing chants
    Returns
    -------
    train_nb_accuracy : float
        accuracy of naive bayes model on modes of training chants
    train_nb_f1 : float
        f1 of naive bayes model on modes of training chants
    test_nb_accuracy : float
        accuracy of naive bayes model on modes of testing chants
    test_nb_f1 : float
        f1 of naive bayes model on modes of testing chants
    """
    # prepare data
    train_data = list2string(train_segments)
    test_data = list2string(test_segments)
    # Fit and eval naive bayes
    pipe = get_nb_model()
    pipe.fit(train_data, train_modes)
    # Predict
    train_predictions = pipe.predict(train_data)
    test_predictions = pipe.predict(test_data)
    # Get scores
    train_nb_accuracy = accuracy_score(train_predictions, train_modes)
    train_nb_f1 = f1_score(train_predictions, train_modes, average='weighted')
    test_nb_accuracy = accuracy_score(test_predictions, test_modes)
    test_nb_f1 = f1_score(test_predictions, test_modes, average='weighted')

    return train_nb_accuracy, train_nb_f1, test_nb_accuracy, test_nb_f1