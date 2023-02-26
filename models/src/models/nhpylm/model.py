from src.models.nhpylm.corpus import Dataset
from src.models.nhpylm.npylm import NPYLM
from src.models.nhpylm.sample import Sampler
from src.models.nhpylm.sentence import Sentence
from src.models.nhpylm.definitions import CHPYLM_BETA_STOP, CHPYLM_BETA_PASS

"""
This is the struct that will serve as a container for the whole NHPYLM. it will be serialized after training.
"""
class Model:
    def __init__(self, dataset: "Dataset", max_word_length: int, initial_a=4.0, initial_b=1.0, chpylm_beta_stop=CHPYLM_BETA_STOP, chpylm_beta_pass=CHPYLM_BETA_PASS):
        max_sentence_length = dataset.max_sentence_length
        # The G_0 probability for the character HPYLM, which depends on the number of different characters in the whole corpus.
        chpylm_G_0 = 1.0 / dataset.vocabulary.get_num_characters()

        # Need to do this because `Model` is immutable
        self.npylm: "NPYLM" = NPYLM(max_word_length, max_sentence_length, chpylm_G_0, initial_a, initial_b, chpylm_beta_stop, chpylm_beta_pass)
        self.sampler: "Sampler" = Sampler(self.npylm, max_word_length, max_sentence_length)

    def get_max_word_length(self):
        return self.npylm.max_word_length

    def set_initial_a(self, initial_a: float):
        self.npylm.lambda_a = initial_a
        self.npylm.sample_lambda_with_initial_params()

    def set_initial_b(self, initial_b: float):
        self.npylm.lambda_b = initial_b
        self.npylm.sample_lambda_with_initial_params()

    def set_chpylm_beta_stop(self, beta_stop:float):
        self.npylm.chpylm.beta_stop = beta_stop

    def set_chpylm_beta_pass(self, beta_pass:float):
        self.npylm.chpylm.beta_pass = beta_pass

    def segment_sentence(self, sentence_string: str):
        self.sampler.extend_capacity(self.npylm.max_word_length, len(sentence_string))
        self.npylm.extend_capacity(len(sentence_string))
        segmented_sentence = []
        sentence = Sentence(sentence_string)
        segment_lengths = self.sampler.viterbi_decode(sentence)

        # I don't even think there's the need to run this method anyways, since all we need is the vector of words eventually.
        # split_sentence(sentence, segment_lengths)

        # Skip the first two BOS in the sentence.
        # start_index = 3
        start_index = 0
        for length in segment_lengths:
            word = sentence_string[int(start_index):int(start_index) + int(length)]
            segmented_sentence.append(word)
            start_index += length

        return ' '.join(segmented_sentence)

    def compute_log_forward_probability(self, sentence_string: str, with_scaling: bool):
        """
        Compute the log forward probability of any sentence given the whole NHPYLM model
        """
        self.sampler.extend_capacity(self.npylm.max_word_length, len(sentence_string))
        self.npylm.extend_capacity(self.npylm, len(sentence_string))
        sentence = Sentence(sentence_string)
        return self.sampler.compute_log_forward_probability(sentence, with_scaling)