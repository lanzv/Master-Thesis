from nhpylm.viterbi_algorithm cimport viterbi_segment_chants
from nhpylm.blocked_gibbs_sampler cimport blocked_gibbs_iteration
from nhpylm.hyperparameters cimport apply_hyperparameters_learning
from nhpylm.npylm cimport NPYLM
from nhpylm.chant cimport Chant
from libc.math cimport exp
import pickle
import lzma
from src.utils.statistics import IterationStatistics
from src.eval.mjww_score import mjwp_score

cdef class NHPYLMModel:
    cdef int max_segment_size
    cdef int n_gram
    cdef float init_d
    cdef float init_theta
    cdef float init_a
    cdef float init_b
    cdef float beta_stops
    cdef float beta_passes
    cdef float d_a
    cdef float d_b
    cdef float theta_alpha
    cdef float theta_beta
    cdef dict train_statistics, dev_statistics
    cdef NPYLM npylm

    
    def __init__(self, max_segment_size = 7, n_gram = 2,
                init_d = 0.5, init_theta = 2.0,
                init_a = 1, init_b = 1,
                beta_stops = 0.57, beta_passes = 0.85,
                d_a = 1, d_b = 1, theta_alpha = 1, theta_beta = 1):
        """
        Init Nested Hirearchical Pitman-Yor Language Model for Gregorian Chants
        TODo
        Parameters
        ----------
        max_segment_size : int
            maximum segment size that the model considers
        n_gram : int
            the n gram of word hirearchical pitman your language model, for now we only support bigrams
        init_d : float
            discount factor in G ~ PY(G0, d, theta) - used in probability recursive computation
        init_theta : float
            theta that controls the average similarity of G and G0 in G ~ PY(G0, d, theta)  - used in probability recrusive computation
        init_a : float
            initial 'a' argument in gamma distribution G(a, b) in order to estimate lambda for poisson correction
        init_b : float
            initial 'b' argument in gamma distribution G(a, b) in order to estimate lambda for poisson correction
        beta_stops : float
            stops hyperparameter, that will be used during sampling and probability computation in THPYLM
            that helps us avoid the situation of zero stops in depth that we want to include
        beta_passes : float
            stops hyperparameter, that will be used during sampling and probability computation in THPYLM
            that helps us to avoid the situation of zero passes in depth that we want to include
        """
        self.max_segment_size = max_segment_size
        self.n_gram = n_gram
        if not n_gram == 2:
            raise NotImplementedError("For now, we support only bigrams, but {} gram was given.".format(n_gram))
        self.init_d = init_d
        self.init_theta = init_theta
        self.init_a = init_a
        self.init_b = init_b
        self.beta_stops = beta_stops
        self.beta_passes = beta_passes
        self.d_a = d_a
        self.d_b = d_b
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta


    cpdef void train(self, list train_data, list dev_data, list train_modes, list dev_modes, int epochs):
        """
        Perform the training process of the NHPYLM model. 
        ToDo
        Parameters
        ----------
        training_chants : list of strings
            list of training chants, each represented as a string
        dev_chants : list of strings
            list of dev chants, each represented as a string
        epochs : int
            number of training epochs
        """
        cdef int i
        cdef str chant_str
        cdef list train_chants = []
        cdef list dev_chants = []
        cdef set train_tone_vocabulary = set()
        cdef str tone
        cdef list train_segments, dev_segments
        cdef float train_perplexity, dev_perplexity
        cdef Chant chant
        cdef object statistics = IterationStatistics(train_modes=train_modes, dev_modes=dev_modes)

        # Prepare Chants
        for chant_str in train_data:
            train_chants.append(Chant(chant_str))
            for tone in chant_str:
                train_tone_vocabulary.add(tone)
        for chant_str in dev_data:
            dev_chants.append(Chant(chant_str))

        # Initialize NPYLM and load all training chants to it
        self.npylm = NPYLM(self.max_segment_size, self.init_d, self.init_theta, 
                            self.init_a, self.init_b,
                            train_tone_vocabulary,
                            self.beta_stops, self.beta_passes,
                            self.d_a, self.d_b, self.theta_alpha, self.theta_beta)
        for chant in train_chants:
            self.npylm.add_chant(chant)

        # Training
        for i in range(epochs):
            blocked_gibbs_iteration(self.npylm, train_chants)
            apply_hyperparameters_learning(self.npylm, train_chants)
            
            # Store and print iteration statistics
            train_segments, train_perplexity = self.predict_segments(train_data)
            dev_segments, dev_perplexity = self.predict_segments(dev_data)
            statistics.add_new_iteration(i+1, train_segments, dev_segments, train_perplexity, dev_perplexity)

        # plot iteration statistics charts and store the model
        statistics.plot_all_statistics()


    cpdef tuple predict_segments(self, list chants):
        """
        Segment chants using this trained model.
        ToDo
        Parameters
        ----------
        chants : list of strings
            list of training chants, each represented as a string
        Returns
        -------
        segmented_chants : list of lists of strings
            list of chants, each chant is represented as list of strings
        """
        cdef float perplexity
        cdef float prob_sum = 0.0
        cdef list chant_segmentation
        cdef list segmented_chants = viterbi_segment_chants(self.npylm, chants)
        for chant_segmentation in segmented_chants:
            prob_sum += self.npylm.get_segmentation_log_probability(chant_segmentation)/len(chant_segmentation)
        perplexity = exp(-prob_sum/len(segmented_chants))
        return segmented_chants, perplexity

    cpdef float get_mjwp_score(self):
        """
        ToDo
        """
        return mjwp_score(self)