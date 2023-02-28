from src.models.nhpylm.sentence import Sentence
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
This struct keeps track of all sentences from the corpus files, and optionally the "true" segmentations, if any.
"""
class Corpus():
    def __init__(self):

        self.sentence_list: list = [] # Vector{UTF32String}
        self.segmented_word_list: list = [] # Vector{Vector{UTF32String}}

    def add_sentence(self, sentence_string: str):
        """
        Add an individual sentence to the corpus
        """
        self.sentence_list.append(sentence_string)

    def load_corpus(self, chants):
        """
        Read the corpus from an input stream
        """
        # Strips the newline character
        for chant in chants:
            if len(chant) == 0:
                continue
            else:
                self.add_sentence(chant)

    def get_num_sentences(self):
        return len(self.sentence_list)

    def get_num_already_segmented_sentences(self):
        return len(self.segmented_word_list)



"""
This struct holds all the structs related to a session/task, including the vocabulary, the corpus and the sentences produced from the corpus.
"""
class Dataset():
    def __init__(self, corpus: "Corpus", train_proportion: float):
        self.vocabulary = Vocabulary()
        self.corpus = corpus
        # Max allowed sentence length in this dataset
        self.max_sentence_length: int = 0
        # Average sentence length in this dataset
        self.avg_sentence_length: float = 0
        self.num_segmented_words: int = 0
        self.train_sentences: list = [] # Vector{Sentence}
        self.dev_sentences: list = [] # Vector{Sentence}

        corpus_length: int = 0
        sentence_indices = [0 for _ in range(corpus.get_num_sentences())]
        for i in range(corpus.get_num_sentences()):
            sentence_indices[i] = i

        np.random.shuffle(sentence_indices)

        # How much of the input data will be used for training vs. used as dev (is there even a dev set in tihs one?)
        train_proportion = min(1.0, max(0.0, train_proportion))
        num_train_sentences = float(corpus.get_num_sentences()) * train_proportion
        for i in range(corpus.get_num_sentences()):
            sentence_string = corpus.sentence_list[sentence_indices[i]]
            if i <= num_train_sentences:
                self.add_sentence(sentence_string, self.train_sentences)
            else:
                self.add_sentence(sentence_string, self.dev_sentences)


            if len(sentence_string) > self.max_sentence_length:
                self.max_sentence_length = len(sentence_string)

            corpus_length += len(sentence_string)

        self.avg_sentence_length = corpus_length / corpus.get_num_sentences()


    def get_num_train_sentences(self):
        return len(self.train_sentences)

    def get_num_dev_sentences(self):
        return len(self.dev_sentences)

    def add_sentence(self, sentence_string: str, sentences: list):
        """
        Add a sentence to the train or dev sentence vector of the dataset
        """
        assert(len(sentence_string) > 0)
        for char in sentence_string:
            self.vocabulary.add_character(char)
        sentences.append(Sentence(sentence_string))