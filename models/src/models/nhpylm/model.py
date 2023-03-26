from src.models.nhpylm.corpus import Dataset
from src.models.nhpylm.npylm import NPYLM
from src.models.nhpylm.sample import Sampler
from src.models.nhpylm.chant import Chant
from src.models.nhpylm.trainer import Trainer
from src.models.nhpylm.definitions import HPYLM_A, HPYLM_B, CHPYLM_BETA_STOP, CHPYLM_BETA_PASS
import pickle
import lzma

"""
This is the struct that will serve as a container for the whole NHPYLM. it will be serialized after training.
"""
class Model:
    def __init__(self, dataset: "Dataset", min_word_length: int, max_word_length: int, initial_a=4.0, initial_b=1.0, chpylm_beta_stop=CHPYLM_BETA_STOP, chpylm_beta_pass=CHPYLM_BETA_PASS, n_gram: int = 3):
        max_chant_length = dataset.max_chant_length
        # The G_0 probability for the character HPYLM, which depends on the number of different characters in the whole corpus.
        chpylm_G_0 = 1.0 / dataset.vocabulary.get_num_characters()

        # Need to do this because `Model` is immutable
        self.npylm: "NPYLM" = NPYLM(min_word_length, max_word_length, max_chant_length, chpylm_G_0, initial_a, initial_b, chpylm_beta_stop, chpylm_beta_pass, n_gram = n_gram)
        self.sampler: "Sampler" = Sampler(self.npylm, min_word_length, max_word_length, max_chant_length, n_gram = n_gram)

        self.dataset = dataset
        self.set_initial_a(HPYLM_A)
        self.set_initial_b(HPYLM_B)
        self.set_chpylm_beta_stop(CHPYLM_BETA_STOP)
        self.set_chpylm_beta_pass(CHPYLM_BETA_PASS)
        self.iterations = 0
        self.trainer = None

    def train(self, epochs = 20):
        if self.trainer == None:
            self.trainer = Trainer(self.dataset, self)

        for epoch in range(1, epochs + 1):
            self.trainer.blocked_gibbs_sampling()
            self.trainer.sample_hyperparameters()
            self.trainer.sample_lambda()
            # The accuracy is better after several iterations have been already done.
            if self.iterations > 2:
                self.trainer.update_p_k_given_chpylm()
            print("Iteration {}".format(self.iterations+1))
            if epoch % 10 == 0:
                self.trainer.print_segmentations_train(10)
                print("Perplexity_dev: {}".format(self.trainer.compute_perplexity_dev()))
            self.iterations += 1
            self.save_model()


    def get_max_word_length(self):
        return self.npylm.max_word_length

    def get_min_word_length(self):
        return self.npylm.min_word_length

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

    def predict_segments(self, chants):
        final_segmentation = []
        for i, chant in enumerate(chants):
            if i%1000 == 0:
                print("Already segmented {} chants".format(i))
            final_segmentation.append((self.segment_chant(chant)).split(' '))
        return final_segmentation

    def segment_chant(self, chant_string: str):
        self.sampler.extend_capacity(self.npylm.max_word_length, len(chant_string))
        self.npylm.extend_capacity(len(chant_string))
        segmented_chant = []
        chant = Chant(chant_string)
        segment_lengths = self.sampler.viterbi_decode(chant)

        # I don't even think there's the need to run this method anyways, since all we need is the vector of words eventually.
        # split_chant(chant, segment_lengths)

        # Skip the first two BOS in the chant.
        # start_index = 3
        start_index = 0
        for length in segment_lengths:
            word = chant_string[int(start_index):int(start_index) + int(length)]
            segmented_chant.append(word)
            start_index += length

        return ' '.join(segmented_chant)

    def compute_log_forward_probability(self, chant_string: str, with_scaling: bool):
        """
        Compute the log forward probability of any chant given the whole NHPYLM model
        """
        self.sampler.extend_capacity(self.npylm.max_word_length, len(chant_string))
        self.npylm.extend_capacity(self.npylm, len(chant_string))
        chant = Chant(chant_string)
        return self.sampler.compute_log_forward_probability(chant, with_scaling)

    def save_model(self, model_path = "./nhpylm_models/model.nhpylm"):
        with lzma.open(model_path, "wb") as model_file:
            pickle.dump(self, model_file)

    def load_model(model_path = "./nhpylm_models/model.nhpylm") -> "Model":
        with lzma.open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        return model