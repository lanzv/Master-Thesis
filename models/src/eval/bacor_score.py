import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import loguniform
import numpy as np
import logging
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from src.utils.eval_helpers import list2string, get_topmelodies_frequency

class _BacorModel:
    def run(self, X_train, y_train, X_test, y_test,
            n_iter=100, n_splits=5):
        train_data, train_targets = X_train, y_train
        test_data, test_targets = X_test, y_test

        # Linear SVC
        svc_params = {
            'penalty': 'l2',
            'loss': 'squared_hinge',
            'multi_class': 'ovr',
            'random_state': np.random.randint(100)
        }
        tuned_params = {
            'clf__C': loguniform(1e-3, 1e4),
            'clf__dual': [True, False]
        }
        self.model = Pipeline([
            ('vect', self.__get_vectorizer()),
            ('clf', LinearSVC(**svc_params)),
        ])

        # Train the model
        return self.__train_model(
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            param_grid=tuned_params,
            n_splits=n_splits,
            n_iter=n_iter
        )

    def evaluate(self, train_pred, train_gold, test_pred, test_gold):
        scores = {"train": {}, "test": {}}
        scores["train"] = dict(
            accuracy = accuracy_score(train_gold, train_pred),
            f1 = f1_score(train_gold, train_pred, average='weighted'),
            precision = precision_score(train_gold, train_pred, average='weighted'),
            recall = recall_score(train_gold, train_pred, average='weighted'),
        )
        scores["test"] = dict(
            accuracy = accuracy_score(test_gold, test_pred),
            f1 = f1_score(test_gold, test_pred, average='weighted'),
            precision = precision_score(test_gold, test_pred, average='weighted'),
            recall = recall_score(test_gold, test_pred, average='weighted'),
        )
        return scores

    def feature_selection(self, X_data, max_features = None):
        selector = SelectFromModel(estimator=self.model.steps[1][1], prefit = True, max_features = max_features)
        X_vectorized = self.model.steps[0][1].transform(X_data)
        logging.info("Features shape: {}".format(X_vectorized.shape))
        X_new = selector.transform(X_vectorized)
        logging.info("Selected features shape: {}".format(X_new.shape))
        chants = self.model.steps[0][1].inverse_transform(X_new)
        melodies = set()
        for chant in chants:
             for melody in chant:
                melodies.add(melody)
        return list(melodies)


    # ------------------------- Training methods -------------------------


    def __train_model(self,
        train_data, train_targets, test_data,
        param_grid, n_splits, n_iter):
        # Tune the model
        self.__tune_model(
            data=train_data,
            targets=train_targets,
            param_grid=param_grid,
            n_splits=n_splits,
            n_iter=n_iter)
        # Train the model
        self.model.fit(train_data, train_targets)

        # Predictions
        train_pred = self.model.predict(train_data)
        test_pred = self.model.predict(test_data)
        return train_pred, test_pred


    def __tune_model(self, data, targets, param_grid, n_splits, n_iter):
        rs = np.random.randint(100)

        # Tune!
        tuner = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
            cv=StratifiedKFold(n_splits=n_splits),
            refit=False,
            n_iter=n_iter,
            n_jobs=-1,
            return_train_score=True,
            random_state=rs)

        tuner.fit(data, targets)
        cv_results = pd.DataFrame(tuner.cv_results_).sort_values('rank_test_accuracy')
        cv_results.index.name = 'id'

        # Log The best results
        best = cv_results.iloc[0, :]

        # Update model parameters
        self.model.set_params(**best.params)

        return cv_results

    def __get_vectorizer(self):
        """Get the standard tfidf-vectorizer"""
        tfidf_params = dict(
            # Defaults
            strip_accents=None,
            stop_words=None,
            ngram_range=(1,1),
            max_df=1.0,
            min_df=1,
            max_features=5000,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            lowercase=False,
            analyzer='word',
            token_pattern=r'[^ ]+')
        vectorizer = TfidfVectorizer(**tfidf_params)
        return vectorizer




def bacor_score(train_segmented_chants, train_modes,
               test_segmented_chants, test_modes, seed = 0, max_features = 100):
    model = _BacorModel()
    # set seed
    np.random.seed(seed)

    # prepare data
    train_data = list2string(train_segmented_chants)
    test_data = list2string(test_segmented_chants)

    # train
    train_pred, test_pred = model.run(X_train = train_data, y_train = train_modes,
                                       X_test = test_data, y_test = test_modes)
    logging.info("The SVC model was trained with {} training data and {} testing data."
        .format(len(train_modes), len(test_modes)))

    # evaluate
    scores = model.evaluate(train_pred, train_modes, test_pred, test_modes)

    # feature selection
    important_melodies = model.feature_selection(train_data, max_features = max_features)

    # plot melody-mode frequencies
    melody_mode_frequencies = get_topmelodies_frequency(
        train_segmented_chants, train_modes, test_segmented_chants, test_modes,
        important_melodies
    )

    return scores, important_melodies, melody_mode_frequencies, model.model