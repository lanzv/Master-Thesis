from src.models.nhpylm.chant import Chant
import numpy as np

"""
This struct keeps track of all the characters in the target corpus.
This is necessary because in the character CHPYLM, the G_0 needs to be calculated via a uniform distribution over all possible characters of the target language.
"""
class Vocabulary():
    def __init__(self):
        self.all_characters: set = set()

    def add_character(self, character):
        self.all_characters.add(character)

    def get_num_characters(self):
        return len(self.all_characters)


"""
This struct keeps track of all chants from the corpus files, and optionally the "true" segmentations, if any.
"""
class Corpus():
    def __init__(self):

        self.chant_list: list = [] # Vector{UTF32String}
        self.segmented_word_list: list = [] # Vector{Vector{UTF32String}}

    def add_chant(self, chant_string: str):
        """
        Add an individual sentence to the corpus
        """
        self.chant_list.append(chant_string)

    def load_corpus(self, chants):
        """
        Read the corpus from an input stream
        """
        # Strips the newline character
        for chant in chants:
            if len(chant) == 0:
                continue
            else:
                self.add_chant(chant)

    def get_num_chants(self):
        return len(self.chant_list)

    def get_num_already_segmented_chants(self):
        return len(self.segmented_word_list)



"""
This struct holds all the structs related to a session/task, including the vocabulary, the corpus and the sentences produced from the corpus.
"""
class Dataset():
    def __init__(self, corpus: "Corpus", train_proportion: float):
        self.vocabulary = Vocabulary()
        self.corpus = corpus
        # Max allowed chant length in this dataset
        self.max_chant_length: int = 0
        # Average chant length in this dataset
        self.avg_chant_length: float = 0
        self.num_segmented_words: int = 0
        self.train_chants: list = [] # Vector{Chant}
        self.dev_chants: list = [] # Vector{Chant}

        corpus_length: int = 0
        chant_indices = [0 for _ in range(corpus.get_num_chants())]
        for i in range(corpus.get_num_chants()):
            chant_indices[i] = i

        np.random.shuffle(chant_indices)

        # How much of the input data will be used for training vs. used as dev (is there even a dev set in tihs one?)
        train_proportion = min(1.0, max(0.0, train_proportion))
        num_train_chants = float(corpus.get_num_chants()) * train_proportion
        for i in range(corpus.get_num_chants()):
            chant_string = corpus.chant_list[chant_indices[i]]
            if i <= num_train_chants:
                self.add_chant(chant_string, self.train_chants)
            else:
                self.add_chant(chant_string, self.dev_chants)


            if len(chant_string) > self.max_chant_length:
                self.max_chant_length = len(chant_string)

            corpus_length += len(chant_string)

        self.avg_chant_length = corpus_length / corpus.get_num_chants()


    def get_num_train_chants(self):
        return len(self.train_chants)

    def get_num_dev_chants(self):
        return len(self.dev_chants)

    def add_chant(self, chant_string: str, chants: list):
        """
        Add a chant to the train or dev chant vector of the dataset
        """
        assert(len(chant_string) > 0)
        for char in chant_string:
            self.vocabulary.add_character(char)
        chants.append(Chant(chant_string))