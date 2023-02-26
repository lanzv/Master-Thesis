from src.models.nhpylm.corpus import Corpus, Dataset
from src.models.nhpylm.model import Model
from src.models.nhpylm.trainer import Trainer
from src.models.nhpylm.definitions import HPYLM_A, HPYLM_B, CHPYLM_BETA_STOP, CHPYLM_BETA_PASS


def build_corpus(path) -> "Corpus":
    corpus = Corpus()
    corpus.read_corpus(path)
    return corpus

def train(corpus_path, split_proportion = 0.9, epochs = 20, max_word_length = 4): # ToDo epochs 1000000
    corpus = build_corpus(corpus_path)
    dataset = Dataset(corpus, split_proportion)

    print("Number of train sentences {}".format(dataset.get_num_train_sentences()))
    print("Number of dev sentences {}".format(dataset.get_num_dev_sentences()))

    vocabulary = dataset.vocabulary
    # Not sure if this is necessary if we already automatically serialize everything.
    # vocab_file = open(joinpath(pwd(), "npylm.dict"))
    # serialize(vocab_file, vocabulary)
    # close(vocab_file)

    model = Model(dataset, max_word_length)
    model.set_initial_a(HPYLM_A)
    model.set_initial_b(HPYLM_B)
    model.set_chpylm_beta_stop(CHPYLM_BETA_STOP)
    model.set_chpylm_beta_pass(CHPYLM_BETA_PASS)

    trainer = Trainer(dataset, model)

    for epoch in range(1, epochs + 1):
        trainer.blocked_gibbs_sampling()
        trainer.sample_hyperparameters()
        trainer.sample_lambda()
        # The accuracy is better after several iterations have been already done.
        if epoch > 3:
            trainer.update_p_k_given_chpylm()
        print("Iteration {}".format(epoch))
        if epoch % 10 == 0:
            trainer.print_segmentations_train(10)
            print("Perplexity_dev: {}".format(trainer.compute_perplexity_dev()))
    return model