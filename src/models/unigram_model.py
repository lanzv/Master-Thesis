import random
import numpy as np
from collections import defaultdict
import logging
from src.utils.loader import load_word_segmentations
from src.models.final_range_classifier import FinalRangeClassifier
from src.eval.maww_score import mawp_score
from src.utils.statistics import IterationStatistics
from math import log, exp

class UnigramModel:
    def __init__(self, min_size = 3, max_size = 8, seed = 0):
        """
        Constructor of the unigram model

        Parameters
        ----------
        min_size: int
            minimal allowed size of segment
        max_size: int
            maximal allowed size of segment
        seed: int
            random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        self.min_size = min_size
        self.max_size = max_size
        self.__init_model()


    def train(self, train_chants, dev_chants, train_modes, dev_modes, init_mode = 'words', iterations = 5, mu = 5, sigma = 2,
                         alpha=0.00000000000001, k_best = 15, print_each = 1):
        """
        The function that trains the mode's variables via Gibbs sampling.
        1. get the init segmentation
        2. iterate the blocked gibbs sampling steps
        3. print training statistics

        Parameters
        ----------
        train_chants: list of strings
            list of string melodies (training chants)
        dev_chants: list of strings
            list of string melodies (dev chants)
        train_modes: list of strings
            list of training modes for evaluations and debugging
        dev_modes: list of strings
            list of dev modes for evaluations and debugging
        init_mode: string
            {words, guassian} inital segmentation
        iterations: int
            number of iterartion steps
        mu: float
            in case of guassian init mode, mu for the Guassian distribution predicting the random init segmentation
        sigma: float
            in case of guassian init mode, sigma for the Guassian distribution predicting the random init segmentation
        alpha: float
            laplace smoothing alpha
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        print_each: int
            interval of printed iterations (evaluated scores)
        """
        # Init model
        self.__init_model()
        self.__generate_vocabulary(train_chants)
        statistics = IterationStatistics(train_modes=train_modes, dev_modes=dev_modes)
        # Do init segmentation, generate model's dictionaries (segment_unigrams, ...)
        if init_mode == 'gaussian':
            init_segmentation = self.__gaus_rand_segments(train_chants, mu, sigma)
        elif init_mode == 'words':
            init_segmentation = self.__word_segments()[:len(train_chants)]
        else:
            raise ValueError("Init mode argument could be only words or gaussian, not {}".format(init_segmentation))
        # Update data structures
        self.chant_count = len(train_chants)
        chant_segmentation = init_segmentation
        for i in range(iterations):
            if i%print_each == 0:
                self.__show_iteration_statistics(statistics, i, train_chants, dev_chants, alpha=alpha, k_best=k_best)
            chant_segmentation = self.__train_iteration(chant_segmentation, k_best = k_best, alpha = alpha)
        
        self.__show_iteration_statistics(statistics, iterations, train_chants, dev_chants, alpha=alpha, k_best=k_best)
        statistics.plot_all_statistics()

    def predict_segments(self, chants, k_best=15, alpha=0.00000000000001):
        """
        Predict chants segmentation.

        Parameters
        ----------
        chants: list of strings
            list of test melodies represented as strings
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        Returns
        -------
        final_segmentation: list of list of strings
            list of segmented chants
        perplexity: float
            perplexity of the prediction
        """
        final_segmentation = []
        log_prob_sum = 0
        for chant_string in chants:
            assert type(chant_string) is str or type(chant_string) is np.str_
            new_segments, chant_prob = self.__get_optimized_chant(chant_segments=[chant_string],
                                                      k_best=k_best, alpha=alpha, argmax=True)
            final_segmentation.append(new_segments)
            log_prob_sum += (chant_prob/len(new_segments))
        perplexity = exp(-log_prob_sum/len(chants))
        return final_segmentation, perplexity

    def get_mawp_score(self):
        """
        Get the mawp score funcion

        Returns
        -------
        mawp_score: float
            mawp score of the model
        """
        return mawp_score(self)

    # ------------------------------- Statistics ----------------------------------
    def __show_iteration_statistics(self, statistics: IterationStatistics, iteration, train_data, dev_data, alpha, k_best):
        """
        Print results of all evaluation functions on train and dev datasets to the given iteration

        Parameters
        ----------
        statistics: IterationStatistics
            IterationStatistics object keeping all statistics and evaluating+printing final string
        iteration: int
            number of iteration
        train_data: list of lists of strings
            list segmented training chants
        dev_data: list of lists of strings
            list of segmented dev chants
        alpha: float
            laplace smoothing alpha
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        """
        train_segments, train_perplexity = self.predict_segments(train_data, k_best=k_best, alpha=alpha)
        dev_segments, dev_perplexity = self.predict_segments(dev_data, k_best=k_best, alpha=alpha)
        statistics.add_new_iteration(iteration, train_segments, dev_segments, train_perplexity, dev_perplexity)

    # ------------------------------- data structures updates -------------------------
    def __init_model(self):
        """
        Initialize model's data structures
        """
        # dictionary of melody string and its counts over all documents (as integer)
        self.segment_unigrams = defaultdict(int)
        # number of all segments, the sum over all counts
        self.total_segments = 0
        # dictionaryof melody strings and its hashset of chants that contains
        # this melody
        self.segment_inverted_index = defaultdict(set)
        # total number of chants
        self.chant_count = 0
        # vocabulary
        self.vocabulary = set()
        


    def __generate_vocabulary(self, chants):
        """
        Generate init vocabulary of all possible segments considering the dataset and allowed segment range

        Parameters
        ----------
        chants: list of strings
            list of string melodies
        """
        self.vocabulary = set()
        for chant_str in chants:
            self.recursion = 0
            self.__update_vocab(chant_str=chant_str, char_id = 0)
        logging.info("Vocabulary was generated with size of {}".format(len(self.vocabulary)))

    def __update_vocab(self,chant_str: str, char_id: int):
        """
        Generate all possible segments of the given chant_str chant and add them into the vocabulary set

        Parameters
        ----------
        chant_str: string
            melody string
        """
        for char_id, c in enumerate(chant_str):
            for segment_boundary_r in range(char_id + self.min_size, 
                                min((char_id+self.max_size+1), len(chant_str)+1)):
                new_segment = chant_str[char_id:segment_boundary_r]
                self.vocabulary.add(new_segment)
                
    def __ignore_chant(self, chant_segments, chant_id):
        """
        Remove the segmented chant from model's datastructures

        Parameters
        ----------
        chant_segments: list of strings
            list of segments, segmented chant
        chant_id:
            id of the chant in the dataset
        """
        for segment in chant_segments:
            # segment unigrams
            self.segment_unigrams[segment] = self.segment_unigrams[segment] - 1
            if self.segment_unigrams[segment] == 0:
                self.segment_unigrams.pop(segment)
            # segment inverted index
            if segment in self.segment_inverted_index and \
            chant_id in self.segment_inverted_index[segment]:
                self.segment_inverted_index[segment].remove(chant_id)
                if len(self.segment_inverted_index[segment]) == 0:
                    self.segment_inverted_index.pop(segment)
        # total segments
        self.total_segments = self.total_segments - len(chant_segments)
        # chant count
        self.chant_count = self.chant_count - 1


    def __add_chant(self, chant_segments, chant_id):
        """
        Add the segmented chant to model's datastructures

        Parameters
        ----------
        chant_segments: list of strings
            list of segments, segmented chant
        chant_id:
            id of the chant in the dataset
        """
        for segment in chant_segments:
            # segment unigrams
            if segment in self.segment_unigrams:
                self.segment_unigrams[segment] = self.segment_unigrams[segment] + 1
            else:
                self.segment_unigrams[segment] = 1
            # segment inverted index
            if segment in self.segment_inverted_index:
                self.segment_inverted_index[segment].add(chant_id)
            else:
                self.segment_inverted_index[segment] = {segment}
        # total segments
        self.total_segments = self.total_segments + len(chant_segments)
        # chant count
        self.chant_count = self.chant_count + 1

    # ------------------------------- init segmentations -----------------------------
    def __gaus_rand_segments(self, chants, mu, sigma):
        """
        Randomly segment chants to get initial gaussian segmentation

        Parameters
        ----------
        chants: list of strings
            list of string melodies
        mu: float
            mu for the Guassian distribution predicting the random init segmentation
        sigma: float
            sigma for the Guassian distribution predicting the random init segmentation
        Returns
        -------
        rand_segmets: list of lists of strings
            list of segmented chants randomly
        """
        rand_segments = []
        for chant_id, chant in enumerate(chants):
            new_chant_segments = []
            i = 0
            while i != len(chant):
                # Find new segment
                new_len = np.clip(a = int(random.gauss(mu, sigma)),
                    a_min = self.min_size, a_max = self.max_size)
                k = min(i+new_len, len(chant))
                new_chant_segments.append(chant[i:k])
                last_added_segment = new_chant_segments[-1]
                # Update segment_unigrams
                if last_added_segment in self.segment_unigrams:
                    self.segment_unigrams[last_added_segment] += 1
                else:
                    self.segment_unigrams[last_added_segment] = 1
                # Update total_segments count
                self.total_segments += 1
                # Update segment_inverted_index
                if last_added_segment in self.segment_inverted_index:
                    self.segment_inverted_index[last_added_segment].add(chant_id)
                else:
                    self.segment_inverted_index[last_added_segment] = {chant_id}
                # Update i index
                i = k
            rand_segments.append(new_chant_segments)
        return rand_segments

    def __word_segments(self):
        """
        Segment chants by words as initial segmentation

        Returns
        -------
        word_segments: list of lists of strings
            list of segmented chants by words
        """
        word_segments = load_word_segmentations()
        for chant_id, chant in enumerate(word_segments):
            for segment in chant:
                # Update segment_unigrams
                if segment in self.segment_unigrams:
                    self.segment_unigrams[segment] += 1
                else:
                    self.segment_unigrams[segment] = 1
                # Update total_segments count
                self.total_segments += 1
                # Update segment_inverted_index
                if segment in self.segment_inverted_index:
                    self.segment_inverted_index[segment].add(chant_id)
                else:
                    self.segment_inverted_index[segment] = {chant_id}
        return word_segments

    # -------------------------------- training ------------------------------
    def __train_iteration(self, segmented_chants, k_best: int, alpha: float):
        """
        Perform the single Gibbs sampling iteration
        1. randomly shuffle dataset indices
        2. iterate over all chants
            - remove segmented chant from model
            - sample chant segmentation
            - add segmented chant to the model

        Parameters
        ----------
        segmented_chants: list of lists of strings
            segmented chants
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        Returns
        -------
        new_segmented_chants: list of lists of strings
            newly segmented chants
        """
        # Gibbs Sampling
        new_segmented_chants = [None for _ in range(len(segmented_chants))]
        rand_indices = np.arange(len(segmented_chants))
        np.random.shuffle(rand_indices)
        for chant_id in rand_indices:
            segments = segmented_chants[chant_id]
            self.__ignore_chant(chant_segments = segments, chant_id=chant_id)
            new_segments, _ = self.__get_optimized_chant(chant_segments=segments,
                                                      k_best=k_best, alpha=alpha)
            self.__add_chant(chant_segments=new_segments, chant_id=chant_id)
            new_segmented_chants[chant_id] = new_segments
        return new_segmented_chants


    def __get_optimized_chant(self, chant_segments, k_best: int, alpha: float, argmax: bool = False):
        """
        Sample the single chant to get new segmentation

        Parameters
        ----------
        chant_segments: list of strings
            list of chant segments
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        argmax: bool
            true for optimizing, false for sampling
        Returns
        -------
        new_segmented_chant: list of strings
            newly segmented chant
        log_prob: float
            log probability of chant segmentation being predicted by the model
        """
        chant = ''.join(chant_segments)
        # for each melody pitch, store the list of k_best nodes (prob, position, prev_node)
        trellis = [[] for _ in range((len(chant)+1))]
        trellis[0] = [Node(0, log(1), None)] # position = 0, prob = 1, prev_node = None
        self.__chant_viterbi_optimization(chant_str=chant, trellis=trellis,
                                          k_best=k_best, alpha=alpha)
        return self.__decode_trellis(trellis, chant, argmax=argmax)



    def __chant_viterbi_optimization(self, chant_str: str, trellis, k_best: int, alpha: float):
        """
        Perform the Viterbi algorithm to build the graph of probabilities

        Parameters
        ----------
        chant_str: string
            chant melody as string
        trellis: list of Nodes
            graph of probabilities
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        """
        for char_id in range(len(chant_str)):
            for segment_boundary_r in range(char_id + self.min_size, 
                                        min((char_id+self.max_size+1), len(chant_str)+1)):
                # update trellis
                new_segment = chant_str[char_id:segment_boundary_r]
                self.__update_trellis(graph=trellis, id=segment_boundary_r,
                                      new_segment=new_segment,
                                      k_best=k_best, alpha=alpha)



    def __update_trellis(self, graph, id: int, new_segment: str, k_best: int, alpha: float):
        """
        Dynamically update probabilities in the trellis graph, always keep the top k paths

        Parameters
        ----------
        graph: list of Nodes
            probability graph of segmentations
        id: int
            node id, or also note id, position of the note
        new_segment: string
            possible segment we are looking on in this update iteration
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        """
        assert len(new_segment) != 0
        prev_id = id - len(new_segment)
        V = len(self.vocabulary)
        new_segment_prob = (self.segment_unigrams[new_segment] + alpha*1)\
                            /(self.total_segments + alpha*V)
        potential_candidates = graph[id]
        for i in range(len(graph[prev_id])):
            new_log_prob = graph[prev_id][i].log_prob + log(new_segment_prob)
            potential_candidates.append(Node(id, new_log_prob, graph[prev_id][i]))
        potential_candidates_sorted = sorted(potential_candidates,
                                             reverse=True, key=lambda x: x.log_prob)

        graph[id] = potential_candidates_sorted[:k_best]



    def __decode_trellis(self, graph: list, chant_str: str, argmax: bool = False):
        """
        Choose the final chant segmentation, compute its log probability

        Parameters
        ----------
        graph: list of Nodes
            list segmentations probabilities
        chant_str: string
            string of melody
        argmax: bool
            true for optimizing, false for sampling
        Returns
        -------
        final_segmentation: list of strings
            newly segmented chant
        log_prob: float
            log probability of chant segmentation being predicted by the model
        """
        final_segmentation_ids = []


        # Choose the final path
        log_probs = np.array([path.log_prob for path in graph[-1]])
        if len(log_probs) == 0:
        # too small chant that is not segmentable into self.min_size,..,self.max_size segments
            return [chant_str], -float('inf')
        if not argmax:
            probs = np.array([exp(log_prob) for log_prob in log_probs])
            if probs.sum() == 0:
            # when probs.sum is too small, face it as a uniform distribution
                probs = np.full((len(probs)), 1/len(probs))
            prob_ind = np.random.choice(
                np.arange(0, len(probs)), p=probs/probs.sum())
            final_node = graph[-1][prob_ind]
            final_log_prob = log_probs[prob_ind]
        else:
            final_node = graph[-1][log_probs.argmax()]
            final_log_prob = log_probs.max()
        final_segmentation_ids.append(final_node.position)


        # find segment ids
        while final_node.prev_node != None:
            final_node = final_node.prev_node
            final_segmentation_ids.append(final_node.position)

        # Final segmentation ids into segmentations
        final_segmentation = []
        for i, j in zip(final_segmentation_ids[:-1], final_segmentation_ids[1:]):
            final_segmentation.append(chant_str[j:i])
        final_segmentation.reverse()
        return final_segmentation, final_log_prob


class UnigramModelModes:
    def __init__(self, min_size = 3, max_size = 8, seed = 0):
        """
        Constructor of the unigram modes model

        Parameters
        ----------
        min_size: int
            minimal allowed size of segment
        max_size: int
            maximal allowed size of segment
        seed: int
            random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        self.min_size = min_size
        self.max_size = max_size
        self.__init_model()

    def train(self, train_chants, dev_chants, train_modes, dev_modes, init_mode = 'words', iterations = 5, mu = 5, sigma = 2,
                         alpha=0.00000000000001, k_best = 15, print_each = 1,
                         final_range_classifier = False, mode_priors_uniform = True):
        """
        The function that trains the mode's variables via Gibbs sampling.
        1. get the init segmentation
        2. iterate the blocked gibbs sampling steps
        3. print training statistics

        Parameters
        ----------
        train_chants: list of strings
            list of string melodies (training chants)
        dev_chants: list of strings
            list of string melodies (dev chants)
        train_modes: list of strings
            list of training modes for evaluations and debugging
        dev_modes: list of strings
            list of dev modes for evaluations and debugging
        init_mode: string
            {words, guassian} inital segmentation
        iterations: int
            number of iterartion steps
        mu: float
            in case of guassian init mode, mu for the Guassian distribution predicting the random init segmentation
        sigma: float
            in case of guassian init mode, sigma for the Guassian distribution predicting the random init segmentation
        alpha: float
            laplace smoothing alpha
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        print_each: int
            interval of printed iterations (evaluated scores)
        final_range_classifier: bool
            true for using the iternal mode classifier being based on the final tone and the range
            false to predict the mode using bayes rule
        mode_priors_uniform: bool
            when final_range_classifier is false:
            true -> mode priors is uniform distribution 1/8
            false -> mode priors is distribution by the training data
        """
        # Init model
        self.__init_model()
        self.__generate_vocabulary(train_chants, train_modes)
        statistics = IterationStatistics(train_modes=train_modes, dev_modes=dev_modes)
        # Do init segmentation, generate model's dictionaries (segment_unigrams, ...)
        if init_mode == 'gaussian':
            init_segmentation = self.__gaus_rand_segments(train_chants, train_modes, mu, sigma)
        elif init_mode == 'words':
            init_segmentation = self.__word_segments(train_modes)[:len(train_chants)]
        else:
            raise ValueError("Init mode argument could be only words or gaussian, not {}".format(init_segmentation))
        chant_segmentation = init_segmentation
        for i in range(iterations):
            if i%print_each == 0:
                self.__show_iteration_statistics(statistics, i, train_chants, dev_chants, final_range_classifier, mode_priors_uniform, k_best = k_best, alpha = alpha)
            chant_segmentation = self.__train_iteration(chant_segmentation, train_modes, k_best = k_best, alpha = alpha)

        self.__show_iteration_statistics(statistics, iterations, train_chants, dev_chants, final_range_classifier, mode_priors_uniform, alpha=alpha, k_best=k_best)
        statistics.plot_all_statistics()

    def predict_segments(self, chants, k_best=15, alpha=0.00000000000001, final_range_classifier = False,
                         mode_priors_uniform = True, modes = None):
        """
        Predict modes and find the chants segmentation

        Parameters
        ----------
        chants: list of strings
            list of test melodies represented as strings
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        final_range_classifier: bool
            true for using the iternal mode classifier being based on the final tone and the range
            false to predict the mode using bayes rule
        mode_priors_uniform: bool
            when final_range_classifier is false:
            true -> mode priors is uniform distribution 1/8
            false -> mode priors is distribution by the training data
        Returns
        -------
        final_segmentation: list of list of strings
            list of segmented chants
        perplexity: float
            perplexity of the prediction
        """
        if modes == None:
            modes = self.predict_modes(chants, final_range_classifier = final_range_classifier,
                                        mode_priors_uniform = mode_priors_uniform, alpha=alpha,
                                        k_best=k_best)
        final_segmentation = []
        log_prob_sum = 0
        for chant_string, mode in zip(chants, modes):
            assert type(chant_string) is str or type(chant_string) is np.str_
            new_segments, chant_prob = self.__get_optimized_chant(chant_segments=[chant_string], mode=mode,
                                                      k_best=k_best, alpha=alpha, argmax=True)
            final_segmentation.append(new_segments)
            log_prob_sum += (chant_prob/len(new_segments))
        perplexity = exp(-log_prob_sum/len(chants))
        return final_segmentation, perplexity

    def predict_modes(self, chants, k_best=15, alpha=0.00000000000001, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"], 
                      final_range_classifier = False, mode_priors_uniform = True):
        """
        Using the Bayes Rule
        p(m|c') = (p(c'|m)*p(m))/(p(c')) ~ p(c'|m)*p(m)  ~ p(c'|m) for constant p(m) and p(c')
        m = argmax p(m|c') ... only when p(m) = 1/8 for all m
        needs to be tested whether p(m) = priors

        Parameters
        ----------
        chants: list of strings
            list of test melodies represented as strings
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        mode_list: list of strings
            list of all mode lables
        final_range_classifier: bool
            true for using the iternal mode classifier being based on the final tone and the range
            false to predict the mode using bayes rule
        mode_priors_uniform: bool
            when final_range_classifier is false:
            true -> mode priors is uniform distribution 1/8
            false -> mode priors is distribution by the training data
        Returns
        -------
        final_modes: list of strings
            list of chants mode prediction
        """
        if final_range_classifier:
            return FinalRangeClassifier.predict(chants=chants), [1 for _ in range(len(chants))]
        else:
            final_modes = []
            training_chants_num = 0
            for mode in self.chant_count:
                training_chants_num += self.chant_count[mode]
            for chant_string in chants:
                chosen_mode = None
                best_prob = -float('inf')
                assert type(chant_string) is str or type(chant_string) is np.str_
                for mode in mode_list:
                    _, chant_log_prob = self.__get_optimized_chant(chant_segments=[chant_string], mode=mode,
                                                            k_best=k_best, alpha=alpha, argmax=True)
                    if mode_priors_uniform:
                        pm = 1.0/float(len(mode_list))
                    else:
                        pm = float(self.chant_count[mode])/float(training_chants_num)
                    if chant_log_prob+log(pm) >= best_prob:
                        chosen_mode = mode
                        best_prob = chant_log_prob+log(pm)
                final_modes.append(chosen_mode)
            return final_modes

    def get_mawp_score(self):
        """
        Get the mawp score funcion

        Returns
        -------
        mawp_score: float
            mawp score of the model
        """
        return mawp_score(self)

    # --------------------------------- printers, plotters ----------------------------
    def __show_iteration_statistics(self, statistics: IterationStatistics, iteration, train_data, dev_data,
                                    final_range_classifier, mode_priors_uniform, k_best, alpha):
        """
        Print results of all evaluation functions on train and dev datasets to the given iteration

        Parameters
        ----------
        statistics: IterationStatistics
            IterationStatistics object keeping all statistics and evaluating+printing final string
        iteration: int
            number of iteration
        train_data: list of lists of strings
            list segmented training chants
        dev_data: list of lists of strings
            list of segmented dev chants
        final_range_classifier: bool
            true for using the iternal mode classifier being based on the final tone and the range
            false to predict the mode using bayes rule
        mode_priors_uniform: bool
            when final_range_classifier is false:
            true -> mode priors is uniform distribution 1/8
            false -> mode priors is distribution by the training data
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        """
        train_segments, train_perplexity = self.predict_segments(train_data, final_range_classifier=final_range_classifier, mode_priors_uniform=mode_priors_uniform, k_best = k_best, alpha = alpha)
        dev_segments, dev_perplexity = self.predict_segments(dev_data, final_range_classifier=final_range_classifier, mode_priors_uniform=mode_priors_uniform, k_best = k_best, alpha = alpha)
        statistics.add_new_iteration(iteration, train_segments, dev_segments, train_perplexity, dev_perplexity)

    # ------------------------------- data structures updates -------------------------
    def __init_model(self, all_modes = ["1", "2", "3", "4", "5", "6", "7", "8"]):
        """
        Initialize model's data structures

        Parameters
        ----------
        all_modes: list of strings
            list of all mode lables
        """
        # dictionary of melody string and its counts over all documents (as integer)
        self.segment_unigrams = {}
        # number of all segments, the sum over all counts
        self.total_segments = {}
        # dictionaryof melody strings and its hashset of chants that contains
        # this melody
        self.segment_inverted_index = {}
        # total number of chants
        self.chant_count = {}
        # vocabulary
        self.vocabulary = {}
        for mode in all_modes:
            self.segment_unigrams[mode] = defaultdict(int)
            self.total_segments[mode] = 0
            self.segment_inverted_index[mode] = defaultdict(set)
            self.chant_count[mode] = 0
            self.vocabulary[mode] = set()

    def __generate_vocabulary(self, chants, modes):
        """
        Generate init vocabulary of all possible segments considering the dataset and allowed segment range

        Parameters
        ----------
        chants: list of strings
            list of string melodies
        modes: list of strings
            list of chants modes
        """
        for chant_str, mode in zip(chants, modes):
            self.__update_vocab(chant_str=chant_str, mode=mode, char_id = 0)

        logging.info("Vocabulary of 1 mode was generated with size of {}".format(len(self.vocabulary["1"])))
        logging.info("Vocabulary of 2 mode was generated with size of {}".format(len(self.vocabulary["2"])))
        logging.info("Vocabulary of 3 mode was generated with size of {}".format(len(self.vocabulary["3"])))
        logging.info("Vocabulary of 4 mode was generated with size of {}".format(len(self.vocabulary["4"])))
        logging.info("Vocabulary of 5 mode was generated with size of {}".format(len(self.vocabulary["5"])))
        logging.info("Vocabulary of 6 mode was generated with size of {}".format(len(self.vocabulary["6"])))
        logging.info("Vocabulary of 7 mode was generated with size of {}".format(len(self.vocabulary["7"])))
        logging.info("Vocabulary of 8 mode was generated with size of {}".format(len(self.vocabulary["8"])))

    def __update_vocab(self,chant_str: str, mode, char_id: int):
        """
        Generate all possible segments of the given chant_str chant and add them into the vocabulary set

        Parameters
        ----------
        chant_str: string
            melody string
        mode: string
            chant's mode
        """
        for char_id, c in enumerate(chant_str):
            for segment_boundary_r in range(char_id + self.min_size, 
                                min((char_id+self.max_size+1), len(chant_str)+1)):
                new_segment = chant_str[char_id:segment_boundary_r]
                self.vocabulary[mode].add(new_segment)
                
    def __ignore_chant(self, chant_segments, mode, chant_id):
        """
        Remove the segmented chant from model's datastructures

        Parameters
        ----------
        chant_segments: list of strings
            list of segments, segmented chant
        chant_id:
            id of the chant in the dataset
        mode: string
            chant's mode
        """
        for segment in chant_segments:
            # segment unigrams
            self.segment_unigrams[mode][segment] = self.segment_unigrams[mode][segment] - 1
            if self.segment_unigrams[mode][segment] == 0:
                self.segment_unigrams[mode].pop(segment)
            # segment inverted index
            if segment in self.segment_inverted_index[mode] and \
            chant_id in self.segment_inverted_index[mode][segment]:
                self.segment_inverted_index[mode][segment].remove(chant_id)
                if len(self.segment_inverted_index[mode][segment]) == 0:
                    self.segment_inverted_index[mode].pop(segment)
        # total segments
        self.total_segments[mode] = self.total_segments[mode] - len(chant_segments)
        # chant count
        self.chant_count[mode] = self.chant_count[mode] - 1


    def __add_chant(self, chant_segments, mode, chant_id):
        """
        Add the segmented chant to model's datastructures

        Parameters
        ----------
        chant_segments: list of strings
            list of segments, segmented chant
        mode: string
            chant's mode
        chant_id:
            id of the chant in the dataset
        """
        for segment in chant_segments:
            # segment unigrams
            if segment in self.segment_unigrams[mode]:
                self.segment_unigrams[mode][segment] = self.segment_unigrams[mode][segment] + 1
            else:
                self.segment_unigrams[mode][segment] = 1
            # segment inverted index
            if segment in self.segment_inverted_index[mode]:
                self.segment_inverted_index[mode][segment].add(chant_id)
            else:
                self.segment_inverted_index[mode][segment] = {segment}
        # total segments
        self.total_segments[mode] = self.total_segments[mode] + len(chant_segments)
        # chant count
        self.chant_count[mode] = self.chant_count[mode] + 1

    # ------------------------------- init segmentations -----------------------------
    def __gaus_rand_segments(self, chants, modes, mu, sigma):
        """
        Randomly segment chants to get initial gaussian segmentation

        Parameters
        ----------
        chants: list of strings
            list of string melodies
        modes: list of strings
            list of chants modes
        mu: float
            mu for the Guassian distribution predicting the random init segmentation
        sigma: float
            sigma for the Guassian distribution predicting the random init segmentation
        Returns
        -------
        rand_segmets: list of lists of strings
            list of segmented chants randomly
        """
        rand_segments = []
        for chant_id, (chant, mode) in enumerate(zip(chants, modes)):
            new_chant_segments = []
            self.chant_count[mode] += 1
            i = 0
            while i != len(chant):
                # Find new segment
                new_len = np.clip(a = int(random.gauss(mu, sigma)),
                    a_min = self.min_size, a_max = self.max_size)
                k = min(i+new_len, len(chant))
                new_chant_segments.append(chant[i:k])
                last_added_segment = new_chant_segments[-1]
                # Update segment_unigrams
                if last_added_segment in self.segment_unigrams[mode]:
                    self.segment_unigrams[mode][last_added_segment] += 1
                else:
                    self.segment_unigrams[mode][last_added_segment] = 1
                # Update total_segments count
                self.total_segments[mode] += 1
                # Update segment_inverted_index
                if last_added_segment in self.segment_inverted_index[mode]:
                    self.segment_inverted_index[mode][last_added_segment].add(chant_id)
                else:
                    self.segment_inverted_index[mode][last_added_segment] = {chant_id}
                # Update i index
                i = k
            rand_segments.append(new_chant_segments)
        return rand_segments

    def __word_segments(self, modes):
        """
        Segment chants by words as initial segmentation

        Parameters
        ----------
        modes: list of strings
            list of chants modes
        Returns
        -------
        word_segments: list of lists of strings
            list of segmented chants by words
        """
        word_segments = load_word_segmentations()[:len(modes)]
        for chant_id, (chant, mode) in enumerate(zip(word_segments, modes)):
            self.chant_count[mode] += 1
            for segment in chant:
                # Update segment_unigrams
                if segment in self.segment_unigrams[mode]:
                    self.segment_unigrams[mode][segment] += 1
                else:
                    self.segment_unigrams[mode][segment] = 1
                # Update total_segments count
                self.total_segments[mode] += 1
                # Update segment_inverted_index
                if segment in self.segment_inverted_index[mode]:
                    self.segment_inverted_index[mode][segment].add(chant_id)
                else:
                    self.segment_inverted_index[mode][segment] = {chant_id}
        return word_segments

    # -------------------------------- training ------------------------------
    def __train_iteration(self, segmented_chants, modes, k_best: int, alpha: float):
        """
        Perform the single Gibbs sampling iteration
        1. randomly shuffle dataset indices
        2. iterate over all chants
            - remove segmented chant from submodel corresponding to the chant's mode
            - sample chant segmentation by the submodel corresponding to the chant's mode
            - add segmented chant to the model to the submodel corresponding to the chant's mode

        Parameters
        ----------
        segmented_chants: list of lists of strings
            segmented chants
        modes: list of strings
            list of chants modes
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        Returns
        -------
        new_segmented_chants: list of lists of strings
            newly segmented chants
        """
        # Gibbs Sampling
        new_segmented_chants = [None for _ in range(len(segmented_chants))]
        rand_indices = np.arange(len(segmented_chants))
        np.random.shuffle(rand_indices)
        for chant_id in rand_indices:
            segments = segmented_chants[chant_id]
            mode = modes[chant_id]
            self.__ignore_chant(chant_segments = segments, mode = mode, chant_id=chant_id)
            new_segments, _ = self.__get_optimized_chant(chant_segments=segments,
                                                      mode=mode,
                                                      k_best=k_best, alpha=alpha)
            self.__add_chant(chant_segments=new_segments, mode=mode, chant_id=chant_id)
            new_segmented_chants[chant_id] = new_segments
        return new_segmented_chants


    def __get_optimized_chant(self, chant_segments, mode, k_best: int, alpha: float, argmax: bool = False):
        """
        Sample the single chant to get new segmentation

        Parameters
        ----------
        chant_segments: list of strings
            list of chant segments
        mode: string
            chant's mode
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        argmax: bool
            true for optimizing, false for sampling
        Returns
        -------
        new_segmented_chant: list of strings
            newly segmented chant
        log_prob: float
            log probability of chant segmentation being predicted by the model
        """
        chant = ''.join(chant_segments)
        # for each melody pitch, store the list of k_best nodes (prob, position, prev_node)
        trellis = [[] for _ in range((len(chant)+1))]
        trellis[0] = [Node(0, log(1), None)] # position = 0, prob = 1, prev_node = None
        self.__chant_viterbi_optimization(chant_str=chant, mode=mode, trellis=trellis,
                                          k_best=k_best, alpha=alpha)
        return self.__decode_trellis(trellis, chant, argmax = argmax, alpha = alpha)



    def __chant_viterbi_optimization(self, chant_str: str, mode, trellis, k_best: int, alpha: float):
        """
        Perform the Viterbi algorithm to build the graph of probabilities

        Parameters
        ----------
        chant_str: string
            chant melody as string
        mode: string
            chant's mode
        trellis: list of Nodes
            graph of probabilities
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        """
        for char_id in range(len(chant_str)):
            for segment_boundary_r in range(char_id + self.min_size, 
                                        min((char_id+self.max_size+1), len(chant_str)+1)):
                # update trellis
                new_segment = chant_str[char_id:segment_boundary_r]
                self.__update_trellis(mode = mode, graph=trellis, id=segment_boundary_r,
                                      new_segment=new_segment,
                                      k_best=k_best, alpha=alpha)



    def __update_trellis(self, mode, graph, id: int, new_segment: str, k_best: int, alpha: float):
        """
        Dynamically update probabilities in the trellis graph, always keep the top k paths

        Parameters
        ----------
        mode: string
            chants' moe
        graph: list of Nodes
            probability graph of segmentations
        id: int
            node id, or also note id, position of the note
        new_segment: string
            possible segment we are looking on in this update iteration
        k_best: int
            number of potentional paths when sampling segmented chants in trainign
        alpha: float
            laplace smoothing alpha
        """
        assert len(new_segment) != 0
        prev_id = id - len(new_segment)
        V = len(self.vocabulary[mode])
        new_segment_prob = (self.segment_unigrams[mode][new_segment] + alpha*1)\
                            /(self.total_segments[mode] + alpha*V)
        potential_candidates = graph[id]
        for i in range(len(graph[prev_id])):
            new_prob = graph[prev_id][i].log_prob + log(new_segment_prob)
            potential_candidates.append(Node(id, new_prob, graph[prev_id][i]))
        potential_candidates_sorted = sorted(potential_candidates,
                                             reverse=True, key=lambda x: x.log_prob)

        graph[id] = potential_candidates_sorted[:k_best]



    def __decode_trellis(self, graph: list, chant_str: str, argmax: bool = False, alpha = 1):
        """
        Choose the final chant segmentation, compute its log probability

        Parameters
        ----------
        graph: list of Nodes
            list segmentations probabilities
        chant_str: string
            string of melody
        argmax: bool
            true for optimizing, false for sampling
        Returns
        -------
        final_segmentation: list of strings
            newly segmented chant
        log_prob: float
            log probability of chant segmentation being predicted by the model
        """
        final_segmentation_ids = []


        # Choose the final path
        log_probs = np.array([path.log_prob for path in graph[-1]])
        if len(log_probs) == 0:
        # too small chant that is not segmentable into self.min_size,..,self.max_size segments
            return [chant_str], -float('inf')
        if not argmax:
            probs = np.array([exp(log_prob) for log_prob in log_probs])
            if probs.sum() == 0:
            # when probs.sum is too small, face it as a uniform distribution
                probs = np.full((len(probs)), 1/len(probs))
            final_node = graph[-1][np.random.choice(
                np.arange(0, len(probs)), p=probs/probs.sum())]
        else:
            final_node = graph[-1][log_probs.argmax()]
        final_segmentation_ids.append(final_node.position)


        # find segment ids
        while final_node.prev_node != None:
            final_node = final_node.prev_node
            final_segmentation_ids.append(final_node.position)

        # Final segmentation ids into segmentations
        final_segmentation = []
        for i, j in zip(final_segmentation_ids[:-1], final_segmentation_ids[1:]):
            final_segmentation.append(chant_str[j:i])
        final_segmentation.reverse()
        final_log_prob = -float('inf')
        for mode in self.vocabulary:
            V = len(self.vocabulary[mode])
            mode_log_prob = log(1)
            for segment in final_segmentation:
                mode_log_prob += log((self.segment_unigrams[mode][segment] + alpha*1)\
                                /(self.total_segments[mode] + alpha*V))
            final_log_prob = np.logaddexp(final_log_prob, mode_log_prob)
            

        return final_segmentation, final_log_prob



class Node:
    def __init__(self, position:int, log_prob:float, prev_node: "Node"):
        self.position = position
        self.log_prob = log_prob
        self.prev_node = prev_node