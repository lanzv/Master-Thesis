import random
import numpy as np
from collections import defaultdict
import logging
from src.utils.loader import load_word_segmentations
from src.eval.pipelines import single_iteration_pipeline
from src.utils.plotters import plot_iteration_statistics
from src.models.final_range_classifier import FinalRangeClassifier

class UnigramModel:
    def __init__(self, min_size = 3, max_size = 8, seed = 0):
        random.seed(seed)
        np.random.seed(seed)
        self.min_size = min_size
        self.max_size = max_size
        self.__init_model()


    def train(self, chants, modes, init_mode = 'words', iterations = 5, mu = 5, sigma = 2,
                         alpha = 1, k_best = 15, print_each = 1, train_proportion: float = 0.9):
        # Divide chants to train and dev datasets
        splitting_point = int(train_proportion*len(chants))
        train_chants, dev_chants = chants[:splitting_point], chants[splitting_point:]
        train_modes, dev_modes = modes[:splitting_point], modes[splitting_point:]
        # Init model
        self.__init_model()
        self.__generate_vocabulary(train_chants)
        # Do init segmentation, generate model's dictionaries (segment_unigrams, ...)
        if init_mode == 'gaussian':
            init_segmentation = self.__gaus_rand_segments(train_chants, mu, sigma)
        elif init_mode == 'words':
            init_segmentation = self.__word_segments()[:splitting_point]
        else:
            raise ValueError("Init mode argument could be only words or gaussian, not {}".format(init_segmentation))
        # Update data structures
        self.chant_count = len(train_chants)
        chant_segmentation = init_segmentation
        for i in range(iterations):
            if i%print_each == 0:
                self.__store_iteration_results(i, train_chants, train_modes, dev_chants, dev_modes)
            chant_segmentation = self.__train_iteration(chant_segmentation, k_best = k_best, alpha = alpha)
        self.__store_iteration_results(iterations, train_chants, train_modes, dev_chants, dev_modes)
        self.__plot_statistics()

    def predict_segments(self, chants, k_best=15, alpha=1):
        final_segmentation = []
        entropy_sum = 0
        for chant_string in chants:
            assert type(chant_string) is str or type(chant_string) is np.str_
            new_segments, chant_prob = self.__get_optimized_chant(chant_segments=[chant_string],
                                                      k_best=k_best, alpha=alpha, argmax=True)
            final_segmentation.append(new_segments)
            if chant_prob > 0:
                entropy_sum -= chant_prob*np.log2(chant_prob)
        perplexity = np.exp2(entropy_sum)
        return final_segmentation, perplexity

    # --------------------------------- printers, plotters ----------------------------
    def __store_iteration_results(self, iteration, train_chants, train_modes, dev_chants, dev_modes):
        top20_melodies = sorted(self.segment_unigrams, key=self.segment_unigrams.get, reverse=True)[:20]
        train_segmentation, _ = self.predict_segments(train_chants)
        dev_segmentation, dev_perplexity = self.predict_segments(dev_chants)
        accuracy, f1, mjww, wtmf, wufpc, vocab_size, avg_segment_len, top_melodies = single_iteration_pipeline(train_segmentation, train_modes, 
                                                                                        dev_segmentation, dev_modes, top20_melodies)
        self.dev_statistics["accuracy"].append(accuracy*100)
        self.dev_statistics["f1"].append(f1*100)
        self.dev_statistics["mjww"].append(mjww*100)
        self.dev_statistics["wtmf"].append(wtmf*100)
        self.dev_statistics["wufpc"].append(wufpc)
        self.dev_statistics["vocab_size"].append(vocab_size)
        self.dev_statistics["avg_segment_len"].append(avg_segment_len)
        self.dev_statistics["perplexity"].append(dev_perplexity)
        self.dev_statistics["iterations"].append(iteration)
        
        print("{}. Iteration \t dev accuracy: {:.2f}%, dev f1: {:.2f}%, dev perplexity {:.6f}, dev vocabulary size: {}, dev avg segment len: {:.2f}, dev mjww: {:.2f}%, dev wtmf: {:.2f}%, dev wufpc: {:.2f} pitches\t\t {}"
            .format(iteration, accuracy*100, f1*100, dev_perplexity, vocab_size, avg_segment_len, mjww*100, wtmf*100, wufpc, top_melodies))

    def __plot_statistics(self):
        statistics_to_plot = {
            "Dev Bacor - not tuned - Accuracy (%)": (self.dev_statistics["iterations"], self.dev_statistics["accuracy"]),
            "Dev Bacor - not tuned - F1 (%)": (self.dev_statistics["iterations"], self.dev_statistics["f1"]),
            "Dev Perplexity": (self.dev_statistics["iterations"], self.dev_statistics["perplexity"]),
            "Dev Vocabulary Size": (self.dev_statistics["iterations"], self.dev_statistics["vocab_size"]),
            "Dev Average Segment Length": (self.dev_statistics["iterations"], self.dev_statistics["avg_segment_len"]),
            "Dev Melody Justified With Words (%)": (self.dev_statistics["iterations"], self.dev_statistics["mjww"]),
            "Dev Weighted Top Mode Frequency (%)": (self.dev_statistics["iterations"], self.dev_statistics["wtmf"]),
            "Dev Weighted Unique Final Pitch Count": (self.dev_statistics["iterations"], self.dev_statistics["wufpc"])
        }
        plot_iteration_statistics(statistics_to_plot)


    # ------------------------------- data structures updates -------------------------
    def __init_model(self):
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
        # Statistics
        self.dev_statistics = {
            "accuracy": [],
            "f1": [],
            "mjww": [],
            "wtmf": [],
            "wufpc": [],
            "vocab_size": [],
            "avg_segment_len": [],
            "perplexity": [],
            "iterations": []
        }

    def __generate_vocabulary(self, chants):
        self.vocabulary = set()
        for chant_str in chants:
            self.recursion = 0
            self.__update_vocab(chant_str=chant_str, char_id = 0)
        logging.info("Vocabulary was generated with size of {}".format(len(self.vocabulary)))

    def __update_vocab(self,chant_str: str, char_id: int):
        for char_id, c in enumerate(chant_str):
            for segment_boundary_r in range(char_id + self.min_size, 
                                min((char_id+self.max_size+1), len(chant_str)+1)):
                new_segment = chant_str[char_id:segment_boundary_r]
                self.vocabulary.add(new_segment)
                
    def __ignore_chant(self, chant_segments, chant_id):
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
        chant = ''.join(chant_segments)
        # for each melody pitch, store the list of k_best nodes (prob, position, prev_node)
        trellis = [[] for _ in range((len(chant)+1))]
        trellis[0] = [Node(0, 1, None)] # position = 0, prob = 1, prev_node = None
        self.__chant_viterbi_optimization(chant_str=chant, trellis=trellis,
                                          k_best=k_best, alpha=alpha)
        return self.__decode_trellis(trellis, chant, argmax=argmax)



    def __chant_viterbi_optimization(self, chant_str: str, trellis, k_best: int, alpha: float):
        for char_id in range(len(chant_str)):
            for segment_boundary_r in range(char_id + self.min_size, 
                                        min((char_id+self.max_size+1), len(chant_str)+1)):
                # update trellis
                new_segment = chant_str[char_id:segment_boundary_r]
                self.__update_trellis(graph=trellis, id=segment_boundary_r,
                                      new_segment=new_segment,
                                      k_best=k_best, alpha=alpha)



    def __update_trellis(self, graph, id: int, new_segment: str, k_best: int, alpha: float):
        assert len(new_segment) != 0
        prev_id = id - len(new_segment)
        V = len(self.vocabulary)
        new_segment_prob = (self.segment_unigrams[new_segment] + alpha*1)\
                            /(self.total_segments + alpha*V)
        potential_candidates = graph[id]
        for i in range(len(graph[prev_id])):
            new_prob = graph[prev_id][i].prob*new_segment_prob
            potential_candidates.append(Node(id, new_prob, graph[prev_id][i]))
        potential_candidates_sorted = sorted(potential_candidates,
                                             reverse=True, key=lambda x: x.prob)

        graph[id] = potential_candidates_sorted[:k_best]



    def __decode_trellis(self, graph: list, chant_str: str, argmax: bool = False):
        final_segmentation_ids = []


        # Choose the final path
        probs = np.array([path.prob for path in graph[-1]])
        if len(probs) == 0:
        # too small chant that is not segmentable into self.min_size,..,self.max_size segments
            return [chant_str]
        if not argmax:
            if probs.sum() == 0:
            # when probs.sum is too small, face it as a uniform distribution
                probs = np.full((len(probs)), 1/len(probs))
            prob_ind = np.random.choice(
                np.arange(0, len(probs)), p=probs/probs.sum())
            final_node = graph[-1][prob_ind]
            final_prob = None # We don't need probability during training
        else:
            final_node = graph[-1][probs.argmax()]
            final_prob = probs.max()
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
        return final_segmentation, final_prob


class UnigramModelModes:
    def __init__(self, min_size = 3, max_size = 8, seed = 0):
        random.seed(seed)
        np.random.seed(seed)
        self.min_size = min_size
        self.max_size = max_size
        self.__init_model()

    def train(self, chants, modes, init_mode = 'words', iterations = 5, mu = 5, sigma = 2,
                         alpha = 1, k_best = 15, print_each = 1, train_proportion: float = 0.9,
                         final_range_classifier = False, mode_priors_uniform = True):
        # Divide chants to train and dev datasets
        splitting_point = int(train_proportion*len(chants))
        train_chants, dev_chants = chants[:splitting_point], chants[splitting_point:]
        train_modes, dev_modes = modes[:splitting_point], modes[splitting_point:]
        # Init model
        self.__init_model()
        self.__generate_vocabulary(train_chants, modes)
        # Do init segmentation, generate model's dictionaries (segment_unigrams, ...)
        if init_mode == 'gaussian':
            init_segmentation = self.__gaus_rand_segments(train_chants, train_modes, mu, sigma)
        elif init_mode == 'words':
            init_segmentation = self.__word_segments(train_modes)
        else:
            raise ValueError("Init mode argument could be only words or gaussian, not {}".format(init_segmentation))
        chant_segmentation = init_segmentation
        for i in range(iterations):
            if i%print_each == 0:
                self.__store_iteration_results(i, train_chants, train_modes, dev_chants, dev_modes,
                                                final_range_classifier, mode_priors_uniform)
            chant_segmentation = self.__train_iteration(chant_segmentation, train_modes, k_best = k_best, alpha = alpha)
        self.__store_iteration_results(iterations, train_chants, train_modes, dev_chants, dev_modes,
                                        final_range_classifier, mode_priors_uniform)
        self.__plot_statistics()

    def predict_segments(self, chants, k_best=15, alpha=1,
                         final_range_classifier = False, mode_priors_uniform = True):
        modes = self.predict_modes(chants, final_range_classifier = final_range_classifier,
                                   mode_priors_uniform = mode_priors_uniform)
        final_segmentation = []
        entropy_sum = 0
        for chant_string, mode in zip(chants, modes):
            assert type(chant_string) is str or type(chant_string) is np.str_
            new_segments, chant_prob = self.__get_optimized_chant(chant_segments=[chant_string], mode=mode,
                                                      k_best=k_best, alpha=alpha, argmax=True)
            final_segmentation.append(new_segments)
            if chant_prob > 0:
                entropy_sum -= chant_prob*np.log2(chant_prob)
        perplexity = np.exp2(entropy_sum)
        return final_segmentation, perplexity

    def predict_modes(self, chants, k_best=15, alpha=1, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"], 
                      final_range_classifier = False, mode_priors_uniform = True):
        """
        Using the Bayes Rule
        p(m|c') = (p(c'|m)*p(m))/(p(c')) ~ p(c'|m)*p(m)  ~ p(c'|m) for constant p(m) and p(c')
        m = argmax p(m|c') ... only when p(m) = 1/8 for all m
        needs to be tested whether p(m) = priors
        """
        if final_range_classifier:
            return FinalRangeClassifier.predict(chants=chants)
        else:
            final_modes = []
            training_chants_num = 0
            for mode in self.chant_count:
                training_chants_num += self.chant_count[mode]
            for chant_string in chants:
                chosen_mode = None
                best_prob = -1
                assert type(chant_string) is str or type(chant_string) is np.str_
                for mode in mode_list:
                    _, chant_prob = self.__get_optimized_chant(chant_segments=[chant_string], mode=mode,
                                                            k_best=k_best, alpha=alpha, argmax=True)
                    if mode_priors_uniform:
                        pm = 1.0/float(len(mode_list))
                    else:
                        pm = float(self.chant_count[mode])/float(training_chants_num)
                    if chant_prob*pm > best_prob:
                        chosen_mode = mode
                        best_prob = chant_prob*pm
                final_modes.append(chosen_mode)
            return final_modes



    # --------------------------------- printers, plotters ----------------------------
    def __store_iteration_results(self, iteration, train_chants, train_modes, dev_chants, dev_modes,
                                  final_range_classifier=False, mode_priors_uniform = True):
        all_melodies = {}
        for mode in self.segment_unigrams:
            for segment in self.segment_unigrams[mode]:
                if segment in all_melodies:
                    all_melodies[segment] += 1
                else:
                    all_melodies[segment] = 1
        top20_melodies = sorted(all_melodies, key=all_melodies.get, reverse=True)[:20]
        train_segmentation, _ = self.predict_segments(train_chants,
                                        final_range_classifier = final_range_classifier,
                                        mode_priors_uniform = mode_priors_uniform)
        dev_segmentation, dev_perplexity = self.predict_segments(dev_chants,
                                        final_range_classifier = final_range_classifier,
                                        mode_priors_uniform = mode_priors_uniform)
        accuracy, f1, mjww, wtmf, wufpc, vocab_size, avg_segment_len, top_melodies = single_iteration_pipeline(train_segmentation, train_modes, 
                                                                                        dev_segmentation, dev_modes, top20_melodies)
        self.dev_statistics["accuracy"].append(accuracy*100)
        self.dev_statistics["f1"].append(f1*100)
        self.dev_statistics["mjww"].append(mjww*100)
        self.dev_statistics["wtmf"].append(wtmf*100)
        self.dev_statistics["wufpc"].append(wufpc)
        self.dev_statistics["vocab_size"].append(vocab_size)
        self.dev_statistics["avg_segment_len"].append(avg_segment_len)
        self.dev_statistics["perplexity"].append(dev_perplexity)
        self.dev_statistics["iterations"].append(iteration)

        print("{}. Iteration \t dev accuracy: {:.2f}%, dev f1: {:.2f}%, dev perplexity {:.6f}, dev vocabulary size: {}, dev avg segment len: {:.2f}, dev mjww: {:.2f}%, dev wtmf: {:.2f}%, dev wufpc: {:.2f} pitches\t\t {}"
            .format(iteration, accuracy*100, f1*100, dev_perplexity, vocab_size, avg_segment_len, mjww*100, wtmf*100, wufpc, top_melodies))


    def __plot_statistics(self):
        statistics_to_plot = {
            "Dev Bacor - not tuned - Accuracy (%)": (self.dev_statistics["iterations"], self.dev_statistics["accuracy"]),
            "Dev Bacor - not tuned - F1 (%)": (self.dev_statistics["iterations"], self.dev_statistics["f1"]),
            "Dev Perplexity": (self.dev_statistics["iterations"], self.dev_statistics["perplexity"]),
            "Dev Vocabulary Size": (self.dev_statistics["iterations"], self.dev_statistics["vocab_size"]),
            "Dev Average Segment Length": (self.dev_statistics["iterations"], self.dev_statistics["avg_segment_len"]),
            "Dev Melody Justified With Words (%)": (self.dev_statistics["iterations"], self.dev_statistics["mjww"]),
            "Dev Weighted Top Mode Frequency (%)": (self.dev_statistics["iterations"], self.dev_statistics["wtmf"]),
            "Dev Weighted Unique Final Pitch Count": (self.dev_statistics["iterations"], self.dev_statistics["wufpc"])
        }
        plot_iteration_statistics(statistics_to_plot)

    # ------------------------------- data structures updates -------------------------
    def __init_model(self, all_modes = ["1", "2", "3", "4", "5", "6", "7", "8"]):
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
        # Statistics
        self.dev_statistics = {
            "accuracy": [],
            "f1": [],
            "mjww": [],
            "wtmf": [],
            "wufpc": [],
            "vocab_size": [],
            "avg_segment_len": [],
            "perplexity": [],
            "iterations": []
        }
        for mode in all_modes:
            self.segment_unigrams[mode] = defaultdict(int)
            self.total_segments[mode] = 0
            self.segment_inverted_index[mode] = defaultdict(set)
            self.chant_count[mode] = 0
            self.vocabulary[mode] = set()

    def __generate_vocabulary(self, chants, modes):
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
        for char_id, c in enumerate(chant_str):
            for segment_boundary_r in range(char_id + self.min_size, 
                                min((char_id+self.max_size+1), len(chant_str)+1)):
                new_segment = chant_str[char_id:segment_boundary_r]
                self.vocabulary[mode].add(new_segment)
                
    def __ignore_chant(self, chant_segments, mode, chant_id):
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
        chant = ''.join(chant_segments)
        # for each melody pitch, store the list of k_best nodes (prob, position, prev_node)
        trellis = [[] for _ in range((len(chant)+1))]
        trellis[0] = [Node(0, 1, None)] # position = 0, prob = 1, prev_node = None
        self.__chant_viterbi_optimization(chant_str=chant, mode=mode, trellis=trellis,
                                          k_best=k_best, alpha=alpha)
        return self.__decode_trellis(trellis, chant, argmax = argmax)



    def __chant_viterbi_optimization(self, chant_str: str, mode, trellis, k_best: int, alpha: float):
        for char_id in range(len(chant_str)):
            for segment_boundary_r in range(char_id + self.min_size, 
                                        min((char_id+self.max_size+1), len(chant_str)+1)):
                # update trellis
                new_segment = chant_str[char_id:segment_boundary_r]
                self.__update_trellis(mode = mode, graph=trellis, id=segment_boundary_r,
                                      new_segment=new_segment,
                                      k_best=k_best, alpha=alpha)



    def __update_trellis(self, mode, graph, id: int, new_segment: str, k_best: int, alpha: float):
        assert len(new_segment) != 0
        prev_id = id - len(new_segment)
        V = len(self.vocabulary[mode])
        new_segment_prob = (self.segment_unigrams[mode][new_segment] + alpha*1)\
                            /(self.total_segments[mode] + alpha*V)
        potential_candidates = graph[id]
        for i in range(len(graph[prev_id])):
            new_prob = graph[prev_id][i].prob*new_segment_prob
            potential_candidates.append(Node(id, new_prob, graph[prev_id][i]))
        potential_candidates_sorted = sorted(potential_candidates,
                                             reverse=True, key=lambda x: x.prob)

        graph[id] = potential_candidates_sorted[:k_best]



    def __decode_trellis(self, graph: list, chant_str: str, argmax: bool = False):
        final_segmentation_ids = []


        # Choose the final path
        probs = np.array([path.prob for path in graph[-1]])
        if len(probs) == 0:
        # too small chant that is not segmentable into self.min_size,..,self.max_size segments
            return [chant_str]
        if not argmax:
            if probs.sum() == 0:
            # when probs.sum is too small, face it as a uniform distribution
                probs = np.full((len(probs)), 1/len(probs))
            final_node = graph[-1][np.random.choice(
                np.arange(0, len(probs)), p=probs/probs.sum())]
            final_prob = None
        else:
            final_node = graph[-1][probs.argmax()]
            final_prob = probs.max()
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
        return final_segmentation, final_prob



class UnigramModel4Modes(UnigramModelModes):
    mode_mapper = {
        "1": "1,2",
        "2": "1,2",
        "3": "3,4",
        "4": "3,4",
        "5": "5,6",
        "6": "5,6",
        "7": "7,8",
        "8": "7,8",
        "1,2": "1,2",
        "3,4": "3,4",
        "5,6": "5,6",
        "7,8": "7,8"
    }
    def __init__(self, min_size = 3, max_size = 8, seed = 0):
        random.seed(seed)
        np.random.seed(seed)
        self.min_size = min_size
        self.max_size = max_size
        self.__init_model()

    def train(self, chants, modes, init_mode = 'words', iterations = 5, mu = 5, sigma = 2,
                         alpha = 1, k_best = 15, print_each = 1, train_proportion: float = 0.9,
                         final_range_classifier = False, mode_priors_uniform = True):
        # Divide chants to train and dev datasets
        splitting_point = int(train_proportion*len(chants))
        train_chants, dev_chants = chants[:splitting_point], chants[splitting_point:]
        train_modes, dev_modes = modes[:splitting_point], modes[splitting_point:]
        # Init model
        self.__init_model()
        self.__generate_vocabulary(train_chants, modes)
        # Do init segmentation, generate model's dictionaries (segment_unigrams, ...)
        if init_mode == 'gaussian':
            init_segmentation = self.__gaus_rand_segments(train_chants, train_modes, mu, sigma)
        elif init_mode == 'words':
            init_segmentation = self.__word_segments(train_modes)
        else:
            raise ValueError("Init mode argument could be only words or gaussian, not {}".format(init_segmentation))
        chant_segmentation = init_segmentation
        for i in range(iterations):
            if i%print_each == 0:
                self.__store_iteration_results(i, train_chants, train_modes, dev_chants, dev_modes,
                                                final_range_classifier, mode_priors_uniform)
            chant_segmentation = self.__train_iteration(chant_segmentation, train_modes, k_best = k_best, alpha = alpha)
        self.__store_iteration_results(iterations, train_chants, train_modes, dev_chants, dev_modes,
                                        final_range_classifier, mode_priors_uniform)
        self.__plot_statistics()

    def predict_segments(self, chants, k_best=15, alpha=1,
                         final_range_classifier = False, mode_priors_uniform = True):
        modes = self.predict_modes(chants, final_range_classifier = final_range_classifier,
                                   mode_priors_uniform = mode_priors_uniform)
        print("new predict segments")
        final_segmentation = []
        entropy_sum = 0
        for chant_string, mode in zip(chants, modes):
            assert type(chant_string) is str or type(chant_string) is np.str_
            new_segments, chant_prob = self.__get_optimized_chant(chant_segments=[chant_string], mode=self.mode_mapper[mode],
                                                      k_best=k_best, alpha=alpha, argmax=True)
            final_segmentation.append(new_segments)
            if chant_prob > 0:
                entropy_sum -= chant_prob*np.log2(chant_prob)
        perplexity = np.exp2(entropy_sum)
        return final_segmentation, perplexity

    def predict_modes(self, chants, k_best=15, alpha=1, mode_list = ["1,2", "3,4", "5,6", "7,8"],
                      final_range_classifier = False, mode_priors_uniform = True):
        print("new predict modes")
        """
        Using the Bayes Rule
        p(m|c') = (p(c'|m)*p(m))/(p(c')) ~ p(c'|m)*p(m)  ~ p(c'|m) for constant p(m) and p(c')
        m = argmax p(m|c') ... only when p(m) = 1/8 for all m
        needs to be tested whether p(m) = priors
        """
        if final_range_classifier:
            return FinalRangeClassifier.predict(chants=chants)
        else:
            final_modes = []
            training_chants_num = 0
            for mode in self.chant_count:
                training_chants_num += self.chant_count[mode]
            for chant_string in chants:
                chosen_mode = None
                best_prob = -1
                assert type(chant_string) is str or type(chant_string) is np.str_
                for mode in mode_list:
                    _, chant_prob = self.__get_optimized_chant(chant_segments=[chant_string], mode=mode,
                                                            k_best=k_best, alpha=alpha, argmax=True)
                    if mode_priors_uniform:
                        pm = 1.0/float(len(mode_list))
                    else:
                        pm = float(self.chant_count[mode])/float(training_chants_num)
                    if chant_prob*pm > best_prob:
                        chosen_mode = mode
                        best_prob = chant_prob*pm
                final_modes.append(chosen_mode)
            return final_modes


    # --------------------------------- printers, plotters ----------------------------
    def __store_iteration_results(self, iteration, train_chants, train_modes, dev_chants, dev_modes,
                                  final_range_classifier=False, mode_priors_uniform = True):
        all_melodies = {}
        for mode in self.segment_unigrams:
            for segment in self.segment_unigrams[mode]:
                if segment in all_melodies:
                    all_melodies[segment] += 1
                else:
                    all_melodies[segment] = 1
        top20_melodies = sorted(all_melodies, key=all_melodies.get, reverse=True)[:20]
        train_segmentation, _ = self.predict_segments(train_chants,
                                        final_range_classifier = final_range_classifier,
                                        mode_priors_uniform = mode_priors_uniform)
        dev_segmentation, dev_perplexity = self.predict_segments(dev_chants,
                                        final_range_classifier = final_range_classifier,
                                        mode_priors_uniform = mode_priors_uniform)
        accuracy, f1, mjww, wtmf, wufpc, vocab_size, avg_segment_len, top_melodies = single_iteration_pipeline(train_segmentation, train_modes, 
                                                                                        dev_segmentation, dev_modes, top20_melodies)
        self.dev_statistics["accuracy"].append(accuracy*100)
        self.dev_statistics["f1"].append(f1*100)
        self.dev_statistics["mjww"].append(mjww*100)
        self.dev_statistics["wtmf"].append(wtmf*100)
        self.dev_statistics["wufpc"].append(wufpc)
        self.dev_statistics["vocab_size"].append(vocab_size)
        self.dev_statistics["avg_segment_len"].append(avg_segment_len)
        self.dev_statistics["perplexity"].append(dev_perplexity)
        self.dev_statistics["iterations"].append(iteration)

        print("{}. Iteration \t dev accuracy: {:.2f}%, dev f1: {:.2f}%, dev perplexity {:.6f}, dev vocabulary size: {}, dev avg segment len: {:.2f}, dev mjww: {:.2f}%, dev wtmf: {:.2f}%, dev wufpc: {:.2f} pitches\t\t {}"
            .format(iteration, accuracy*100, f1*100, dev_perplexity, vocab_size, avg_segment_len, mjww*100, wtmf*100, wufpc, top_melodies))


    def __plot_statistics(self):
        statistics_to_plot = {
            "Dev Bacor - not tuned - Accuracy (%)": (self.dev_statistics["iterations"], self.dev_statistics["accuracy"]),
            "Dev Bacor - not tuned - F1 (%)": (self.dev_statistics["iterations"], self.dev_statistics["f1"]),
            "Dev Perplexity": (self.dev_statistics["iterations"], self.dev_statistics["perplexity"]),
            "Dev Vocabulary Size": (self.dev_statistics["iterations"], self.dev_statistics["vocab_size"]),
            "Dev Average Segment Length": (self.dev_statistics["iterations"], self.dev_statistics["avg_segment_len"]),
            "Dev Melody Justified With Words (%)": (self.dev_statistics["iterations"], self.dev_statistics["mjww"]),
            "Dev Weighted Top Mode Frequency (%)": (self.dev_statistics["iterations"], self.dev_statistics["wtmf"]),
            "Dev Weighted Unique Final Pitch Count": (self.dev_statistics["iterations"], self.dev_statistics["wufpc"])
        }
        plot_iteration_statistics(statistics_to_plot)

    # ------------------------------- data structures updates -------------------------
    def __init_model(self, all_modes = ["1,2", "3,4", "5,6", "7,8"]):
        print("new init model")
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
        # Statistics
        self.dev_statistics = {
            "accuracy": [],
            "f1": [],
            "mjww": [],
            "wtmf": [],
            "wufpc": [],
            "vocab_size": [],
            "avg_segment_len": [],
            "perplexity": [],
            "iterations": []
        }
        for mode in all_modes:
            self.segment_unigrams[mode] = defaultdict(int)
            self.total_segments[mode] = 0
            self.segment_inverted_index[mode] = defaultdict(set)
            self.chant_count[mode] = 0
            self.vocabulary[mode] = set()

    def __generate_vocabulary(self, chants, modes):
        print("new generate vocabulary")
        for chant_str, mode in zip(chants, modes):
            self.__update_vocab(chant_str=chant_str, mode=self.mode_mapper[mode], char_id = 0)

        logging.info("Vocabulary of 1,2 mode was generated with size of {}".format(len(self.vocabulary["1,2"])))
        logging.info("Vocabulary of 3,4 mode was generated with size of {}".format(len(self.vocabulary["3,4"])))
        logging.info("Vocabulary of 5,6 mode was generated with size of {}".format(len(self.vocabulary["5,6"])))
        logging.info("Vocabulary of 7,8 mode was generated with size of {}".format(len(self.vocabulary["7,8"])))

    def __update_vocab(self,chant_str: str, mode, char_id: int):
        for char_id, c in enumerate(chant_str):
            for segment_boundary_r in range(char_id + self.min_size,
                                min((char_id+self.max_size+1), len(chant_str)+1)):
                new_segment = chant_str[char_id:segment_boundary_r]
                self.vocabulary[mode].add(new_segment)

    def __ignore_chant(self, chant_segments, mode, chant_id):
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
        print("new rand segments")
        rand_segments = []
        for chant_id, (chant, mode) in enumerate(zip(chants, modes)):
            new_chant_segments = []
            self.chant_count[self.mode_mapper[mode]] += 1
            i = 0
            while i != len(chant):
                # Find new segment
                new_len = np.clip(a = int(random.gauss(mu, sigma)),
                    a_min = self.min_size, a_max = self.max_size)
                k = min(i+new_len, len(chant))
                new_chant_segments.append(chant[i:k])
                last_added_segment = new_chant_segments[-1]
                # Update segment_unigrams
                if last_added_segment in self.segment_unigrams[self.mode_mapper[mode]]:
                    self.segment_unigrams[self.mode_mapper[mode]][last_added_segment] += 1
                else:
                    self.segment_unigrams[self.mode_mapper[mode]][last_added_segment] = 1
                # Update total_segments count
                self.total_segments[self.mode_mapper[mode]] += 1
                # Update segment_inverted_index
                if last_added_segment in self.segment_inverted_index[self.mode_mapper[mode]]:
                    self.segment_inverted_index[self.mode_mapper[mode]][last_added_segment].add(chant_id)
                else:
                    self.segment_inverted_index[self.mode_mapper[mode]][last_added_segment] = {chant_id}
                # Update i index
                i = k
            rand_segments.append(new_chant_segments)
        return rand_segments

    def __word_segments(self, modes):
        print("new word segments")
        word_segments = load_word_segmentations()[:len(modes)]
        for chant_id, (chant, mode) in enumerate(zip(word_segments, modes)):
            self.chant_count[self.mode_mapper[mode]] += 1
            for segment in chant:
                # Update segment_unigrams
                if segment in self.segment_unigrams[self.mode_mapper[mode]]:
                    self.segment_unigrams[self.mode_mapper[mode]][segment] += 1
                else:
                    self.segment_unigrams[self.mode_mapper[mode]][segment] = 1
                # Update total_segments count
                self.total_segments[self.mode_mapper[mode]] += 1
                # Update segment_inverted_index
                if segment in self.segment_inverted_index[self.mode_mapper[mode]]:
                    self.segment_inverted_index[self.mode_mapper[mode]][segment].add(chant_id)
                else:
                    self.segment_inverted_index[self.mode_mapper[mode]][segment] = {chant_id}
        return word_segments

    # -------------------------------- training ------------------------------
    def __train_iteration(self, segmented_chants, modes, k_best: int, alpha: float):
        print("new train iteration")
        # Gibbs Sampling
        new_segmented_chants = [None for _ in range(len(segmented_chants))]
        rand_indices = np.arange(len(segmented_chants))
        np.random.shuffle(rand_indices)
        for chant_id in rand_indices:
            segments = segmented_chants[chant_id]
            mode = modes[chant_id]
            self.__ignore_chant(chant_segments = segments, mode = self.mode_mapper[mode], chant_id=chant_id)
            new_segments, _ = self.__get_optimized_chant(chant_segments=segments,
                                                      mode=self.mode_mapper[mode],
                                                      k_best=k_best, alpha=alpha)
            self.__add_chant(chant_segments=new_segments, mode=self.mode_mapper[mode], chant_id=chant_id)
            new_segmented_chants[chant_id] = new_segments
        return new_segmented_chants


    def __get_optimized_chant(self, chant_segments, mode, k_best: int, alpha: float, argmax: bool = False):
        chant = ''.join(chant_segments)
        # for each melody pitch, store the list of k_best nodes (prob, position, prev_node)
        trellis = [[] for _ in range((len(chant)+1))]
        trellis[0] = [Node(0, 1, None)] # position = 0, prob = 1, prev_node = None
        self.__chant_viterbi_optimization(chant_str=chant, mode=mode, trellis=trellis,
                                          k_best=k_best, alpha=alpha)
        return self.__decode_trellis(trellis, chant, argmax = argmax)



    def __chant_viterbi_optimization(self, chant_str: str, mode, trellis, k_best: int, alpha: float):
        for char_id in range(len(chant_str)):
            for segment_boundary_r in range(char_id + self.min_size,
                                        min((char_id+self.max_size+1), len(chant_str)+1)):
                # update trellis
                new_segment = chant_str[char_id:segment_boundary_r]
                self.__update_trellis(mode = mode, graph=trellis, id=segment_boundary_r,
                                      new_segment=new_segment,
                                      k_best=k_best, alpha=alpha)



    def __update_trellis(self, mode, graph, id: int, new_segment: str, k_best: int, alpha: float):
        assert len(new_segment) != 0
        prev_id = id - len(new_segment)
        V = len(self.vocabulary[mode])
        new_segment_prob = (self.segment_unigrams[mode][new_segment] + alpha*1)\
                            /(self.total_segments[mode] + alpha*V)
        potential_candidates = graph[id]
        for i in range(len(graph[prev_id])):
            new_prob = graph[prev_id][i].prob*new_segment_prob
            potential_candidates.append(Node(id, new_prob, graph[prev_id][i]))
        potential_candidates_sorted = sorted(potential_candidates,
                                             reverse=True, key=lambda x: x.prob)

        graph[id] = potential_candidates_sorted[:k_best]



    def __decode_trellis(self, graph: list, chant_str: str, argmax: bool = False):
        final_segmentation_ids = []


        # Choose the final path
        probs = np.array([path.prob for path in graph[-1]])
        if len(probs) == 0:
        # too small chant that is not segmentable into self.min_size,..,self.max_size segments
            return [chant_str]
        if not argmax:
            if probs.sum() == 0:
            # when probs.sum is too small, face it as a uniform distribution
                probs = np.full((len(probs)), 1/len(probs))
            final_node = graph[-1][np.random.choice(
                np.arange(0, len(probs)), p=probs/probs.sum())]
            final_prob = None
        else:
            final_node = graph[-1][probs.argmax()]
            final_prob = probs.max()
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
        return final_segmentation, final_prob

class Node:
    def __init__(self, position:int, prob:float, prev_node: "Node"):
        self.position = position
        self.prob = prob
        self.prev_node = prev_node