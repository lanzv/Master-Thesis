from nhpylm.viterbi_algorithm cimport viterbi_segment_chants
from nhpylm.blocked_gibbs_sampler cimport blocked_gibbs_iteration
from nhpylm.hyperparameters cimport apply_hyperparameters_learning
from nhpylm.npylm cimport NPYLM
from nhpylm.chant cimport Chant
from libc.math cimport exp
import numpy as np
cimport numpy as np
import logging
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
                init_a = 6.0, init_b = 0.8333,
                beta_stops = 1.0, beta_passes = 1.0,
                d_a = 1.0, d_b = 1.0, theta_alpha = 1.0, theta_beta = 1.0):
        """
        Init Nested Hirearchical Pitman-Yor Language Model for Gregorian Chants
        
        Parameters
        ----------
        max_segment_size : int
            maximum segment size that the model considers
        n_gram : int
            the n gram of word hirearchical pitman your language model, for now we only support bigrams
        init_d : float
            initial discount factor in G ~ PY(G0, d, theta) - used in probability recursive computation
            hyperparameter d is learned during training
        init_theta : float
            initial theta that controls the average similarity of G and G0 in G ~ PY(G0, d, theta) - used in probability recrusive computation
            hyperparameter theta is learned during trainign
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
        d_a : float
            a parameter for d learning as a base of alpha of Beta distribution
        d_b : float
            b parameter for d learning as a base of beta of Beta distribution
        theta_alpha : float
            alpha parameter for theta learning as a base of shape of Gamma distribution
        theta_beta : float
            beta parameter for theta learning as base of scale of Gamma distribution
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
        logging.info("The NHPYLM model was initialized with max segment size {}".format(max_segment_size))


    cpdef void train(self, list train_data, list dev_data, list train_modes, list dev_modes, 
                    int epochs, bint d_theta_learning, bint poisson_learning, int print_each_nth_iteration = 5):
        """
        Perform the training process of the NHPYLM model. Each iteration print current statistics.
        For those statistics, gold modes are used.
        Apply the hyperparameter learning after each iteration. Function parameters specify those learnings.

        Parameters
        ----------
        train_data : list of strings
            list of training chants, each represented as a string
        dev_data : list of strings
            list of dev chants, each represented as a string
        train_modes : list of strings
            list of train modes used for statistics evaluation
        dev_modes : list of strings
            list of dev modes used for statistics evaluation
        epochs : int
            number of training epochs
        d_theta_learning : boolean
            whether we want to apply d theta learning after each epoch or not
        poisson_learning : boolean
            whether we want to apply poisson learning or not
        print_each_nth_iteration : int
            print only iterations modulo print_each_nth_iteration
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
        logging.info("NHPYLM train - {} train chants, {} dev chants.".format(len(train_data), len(dev_data)))
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
            apply_hyperparameters_learning(self.npylm, train_chants, d_theta_learning, poisson_learning)

            if (i+1)%print_each_nth_iteration == 0:
                train_segments, train_perplexity = self.predict_segments(train_data)
                dev_segments, dev_perplexity = self.predict_segments(dev_data)
                statistics.add_new_iteration(i+1, train_segments, dev_segments, train_perplexity, dev_perplexity)

        # plot iteration statistics charts and store the model
        statistics.plot_all_statistics()

    cpdef tuple predict_segments(self, list chants):
        """
        Segment chants using this trained model. Compute perplexity.

        Parameters
        ----------
        chants : list of strings
            list of training chants, each represented as a string
        Returns
        -------
        segmented_chants : list of lists of strings
            list of chants, each chant is represented as list of strings
        perplexity : float
            compute perplexity of segmented chants
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
        Use mjwp_score function to compute mjwp score of this model.

        Returns
        -------
        mjwp_score : float
            melody justified with phrase score
        """
        return mjwp_score(self)

cdef class NHPYLMModesModel:
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
    cdef dict npylm_modes

    
    def __init__(self, max_segment_size = 7, n_gram = 2,
                init_d = 0.5, init_theta = 2.0,
                init_a = 6.0, init_b = 0.8333,
                beta_stops = 1.0, beta_passes = 1.0,
                d_a = 1.0, d_b = 1.0, theta_alpha = 1.0, theta_beta = 1.0):
        """
        Init Nested Hirearchical Pitman-Yor Language Model for Gregorian Chants
        
        Parameters
        ----------
        max_segment_size : int
            maximum segment size that the model considers
        n_gram : int
            the n gram of word hirearchical pitman your language model, for now we only support bigrams
        init_d : float
            initial discount factor in G ~ PY(G0, d, theta) - used in probability recursive computation
            hyperparameter d is learned during training
        init_theta : float
            initial theta that controls the average similarity of G and G0 in G ~ PY(G0, d, theta) - used in probability recrusive computation
            hyperparameter theta is learned during trainign
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
        d_a : float
            a parameter for d learning as a base of alpha of Beta distribution
        d_b : float
            b parameter for d learning as a base of beta of Beta distribution
        theta_alpha : float
            alpha parameter for theta learning as a base of shape of Gamma distribution
        theta_beta : float
            beta parameter for theta learning as base of scale of Gamma distribution
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
        logging.info("The NHPYLM model was initialized with max segment size {}".format(max_segment_size))


    cpdef void train(self, list train_data, list dev_data, list train_modes, list dev_modes, 
                    int epochs, bint d_theta_learning, bint poisson_learning, int print_each_nth_iteration = 5,
                    list mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
        """
        Initialize eight NHPYLMs, one nhpylm model for each mode. Divide data and train NHPYLMs
        Apply the hyperparameter learning after each iteration. Function parameters specify those learnings.

        Parameters
        ----------
        train_data : list of strings
            list of training chants, each represented as a string
        dev_data : list of strings
            list of dev chants, each represented as a string
        train_modes : list of strings
            list of train modes
        dev_modes : list of strings
            list of dev modes
        epochs : int
            number of training epochs
        d_theta_learning : boolean
            whether we want to apply d theta learning after each epoch or not
        poisson_learning : boolean
            whether we want to apply poisson learning or not
        print_each_nth_iteration : int
            print only iterations modulo print_each_nth_iteration
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
        cdef str mode
        logging.info("NHPYLM train - {} train chants, {} dev chants.".format(len(train_data), len(dev_data)))
        cdef object statistics = IterationStatistics(train_modes=train_modes, dev_modes=dev_modes)

        # Prepare Chants
        for chant_str in train_data:
            train_chants.append(Chant(chant_str))
            for tone in chant_str:
                train_tone_vocabulary.add(tone)
        for chant_str in dev_data:
            dev_chants.append(Chant(chant_str))

        # Initialize NPYLM and load all training chants to it
        self.npylm_modes = {}
        for mode in mode_list:
            self.npylm_modes[mode] = NPYLM(self.max_segment_size, self.init_d, self.init_theta, 
                            self.init_a, self.init_b,
                            train_tone_vocabulary,
                            self.beta_stops, self.beta_passes,
                            self.d_a, self.d_b, self.theta_alpha, self.theta_beta)
        cdef dict train_chants_modes = {}
        cdef NPYLM npylm
        for chant, mode in zip(train_chants, train_modes):
            npylm = self.npylm_modes[mode]
            npylm.add_chant(chant)
            if not mode in train_chants_modes:
                train_chants_modes[mode] = []
            train_chants_modes[mode].append(chant)

        # Training
        for i in range(epochs):
            for mode in mode_list:
                blocked_gibbs_iteration(self.npylm_modes[mode], train_chants_modes[mode])
                apply_hyperparameters_learning(self.npylm_modes[mode], train_chants_modes[mode], d_theta_learning, poisson_learning)

            if (i+1)%print_each_nth_iteration == 0:
                train_segments, train_perplexity = self.predict_segments(train_data)
                dev_segments, dev_perplexity = self.predict_segments(dev_data)
                statistics.add_new_iteration(i+1, train_segments, dev_segments, train_perplexity, dev_perplexity)

        # plot iteration statistics charts and store the model
        statistics.plot_all_statistics()

    cpdef tuple predict_segments_modes(self, list chants):
        """
        Call the predict_modes function. Based on its values, compute perplexity.

        Parameters
        ----------
        chants : list of strings
            list of training chants, each represented as a string
        Returns
        -------
        segmented_chants : list of lists of strings
            list of chants, each chant is represented as list of strings
        perplexity : float
            compute perplexity of segmented chants
        modes : list of strings
            list of predicted modes
        """
        cdef list modes
        cdef list segmentations
        cdef list segmentation_log_probs
        modes, segmentations, segmentation_log_probs = self.predict_modes(chants)
        cdef list segmentation
        cdef float log_prob
        cdef float perplexity
        cdef float prob_sum = 0.0
        for segmentation, log_prob in zip(segmentations, segmentation_log_probs):
            prob_sum += (log_prob/len(segmentation))
        perplexity = exp(-prob_sum/len(chants))

        return segmentations, perplexity, modes


    cpdef tuple predict_segments(self, list chants):
        """
        Call the predict_modes function. Based on its values, compute perplexity.

        Parameters
        ----------
        chants : list of strings
            list of training chants, each represented as a string
        Returns
        -------
        segmented_chants : list of lists of strings
            list of chants, each chant is represented as list of strings
        perplexity : float
            compute perplexity of segmented chants
        """
        cdef list segmentations
        cdef list segmentation_log_probs
        _, segmentations, segmentation_log_probs = self.predict_modes(chants)
        cdef list segmentation
        cdef float log_prob
        cdef float perplexity
        cdef float prob_sum = 0.0
        for segmentation, log_prob in zip(segmentations, segmentation_log_probs):
            prob_sum += (log_prob/len(segmentation))
        perplexity = exp(-prob_sum/len(chants))

        return segmentations, perplexity


    cpdef tuple predict_modes(self, list chants, list mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
        """
        Predict modes for comming chants using Bayes rule. Consider all modes and their top segmentations via viterbi algorithm.
        Choose the best one. Compute probs of rest modes for the fixed segmentation. Take argmax of mode for the probability.
        Take the best segmentaiton and sum of the prob segmentation over all modes, which is a probability of chant segmentaion.

        Parameters
        ----------
        chants : list of strings
            list of chants represented as strings of notes
        mode_list : list of string
            list of all modes in dataset
        Returns
        -------
        modes : list of strings
            list of predicted modes
        segmentations : list of lists of strings
            list of segmentations represented as list of string segments
        segmentation_log_probs : list of floats
            list of log probabilities of chosen segmentaitons (sum of segmentation over all modes)
        """
        cdef str mode
        cdef dict segmented_chants_modes = {}
        # Predict segmentations for each modes
        for mode in mode_list:
            segmented_chants_modes[mode] = viterbi_segment_chants(self.npylm_modes[mode], chants)
        
        # Find the best modes and its segmentations
        cdef list chant_segmentation
        cdef list best_modes = []
        cdef str best_mode
        cdef float best_prob
        cdef float prob
        cdef int i
        cdef NPYLM npylm
        for i in range(len(chants)):
            best_mode = ""
            best_prob = -float('inf')
            for mode in mode_list:
                chant_segmentation = segmented_chants_modes[mode][i]
                npylm = self.npylm_modes[mode]
                prob = npylm.get_segmentation_log_probability(chant_segmentation)
                if prob > best_prob:
                    best_mode = mode
                    best_prob = prob
            best_modes.append(best_mode)

        # Check that there is no better mode for the segmentation
        # otherwise rewrite the best mode
        cdef list new_best_modes = []
        cdef str new_best_mode
        for i, best_mode in enumerate(best_modes):
            chant_segmentation = segmented_chants_modes[best_mode][i]
            new_best_mode = ""
            best_prob = -float('inf')
            for mode in mode_list:
                npylm = self.npylm_modes[mode]
                prob = npylm.get_segmentation_log_probability(chant_segmentation)
                if prob > best_prob:
                    new_best_mode = mode
                    best_prob = prob
            new_best_modes.append(new_best_mode)

        # Compute sum of prob of best segmentation over all modes considering logarithm
        cdef list modes = []
        cdef list segmentations = []
        cdef list segmentation_log_probs = []
        cdef float prob_sum
        for i, best_mode in enumerate(new_best_modes):
            modes.append(best_mode)
            chant_segmentation = segmented_chants_modes[best_mode][i]
            segmentations.append(chant_segmentation)
            prob_sum = -float('inf')
            for mode in mode_list:
                npylm = self.npylm_modes[mode]
                prob_sum = np.logaddexp(prob_sum, npylm.get_segmentation_log_probability(chant_segmentation))
            segmentation_log_probs.append(prob_sum)

        return modes, segmentations, segmentation_log_probs
        


    cpdef float get_mjwp_score(self):
        """
        Use mjwp_score function to compute mjwp score of this model.

        Returns
        -------
        mjwp_score : float
            melody justified with phrase score
        """
        return mjwp_score(self)