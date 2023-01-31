import random
import numpy as np
from collections import defaultdict

class NaiveStatisticModel:
    def __init__(self, seed = 0):
        random.seed(seed)
        np.random.seed(seed)

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


    def predict_segments(self, chants, iterations = 5,
                        epsilon = 0.05, mu = 5, sigma = 2, print_each = 1):
        # Do init segmentation, generate model's dictionaries (segment_unigrams, ...)
        init_segmentation = self.__gaus_rand_segments(chants, mu, sigma)
        # Update data structures
        self.chant_count = len(chants)
        self.__update_vocabulary()
        chant_segmentation = init_segmentation
        for i in range(iterations):
            chant_segmentation = self.__train_iteration(chant_segmentation, epsilon)
            if i%print_each == 0:
                print("{}. Iteration".format(i))
                top25_melodies = sorted(self.segment_unigrams, key=self.segment_unigrams.get, reverse=True)[:30]
                print("\t\t\t", top25_melodies)
                #for topmel in top25_melodies:
                #  print("\t\t\t{}".format(topmel))
        return chant_segmentation

    # ------------------------------- data structures updates -------------------------
    def __update_vocabulary(self):
        self.vocabulary = set()
        for segment in self.segment_unigrams:
            if self.segment_unigrams[segment] > 0:
              self.vocabulary.add(segment)

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
        # update vocabulary
        self.__update_vocabulary()


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
        # update vocabulary
        self.__update_vocabulary()

    # ------------------------------- init segmentations -----------------------------
    def __gaus_rand_segments(self, chants, mu, sigma):
        rand_segments = []
        for chant_id, chant in enumerate(chants):
            new_chant_segments = []
            i = 0
            while i != len(chant):
                # Find new segment
                new_len = max(int(random.gauss(mu, sigma)), 1)
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

    # -------------------------------- training ------------------------------
    def __train_iteration(self, segmented_chants, epsilon, limit = 5):
        new_segmented_chants = []
        for chant_id, segments in enumerate(segmented_chants):
            self.__ignore_chant(chant_segments = segments, chant_id = chant_id)
            new_segments = []  # ToDo remove compounding - temporary solution for making the algorithm fast
            for i in range(0, len(segments), limit):
              new_segments += self.__get_optimized_chant(
                  chant_segments = segments[i:i + limit], epsilon = epsilon)
            self.__add_chant(chant_segments = new_segments, chant_id = chant_id)
            new_segmented_chants.append(new_segments)
        return new_segmented_chants


    def __get_optimized_chant(self, chant_segments, epsilon):
        all_candidates = [] # list of all possible segments
        self.__get_all_chant_possibilities(chant_segments = chant_segments,
                                          new_segments = [],
                                          ith_segment = 0,
                                          epsilon = epsilon,
                                          all_candidates = all_candidates)
        candidates_probs = self.__get_candidates_probs(all_candidates)
        candidate_id = np.random.choice(np.arange(0, len(all_candidates)), p=candidates_probs)
        return all_candidates[candidate_id]


    def __get_all_chant_possibilities(self, chant_segments, new_segments,
                                      ith_segment, epsilon, all_candidates = [],
                                      join_segment = None):
        if len(chant_segments) == ith_segment:
            all_candidates.append(new_segments)
        elif join_segment != None:
            joined_segment = join_segment + chant_segments[ith_segment]
            extended_segments = new_segments.copy()
            extended_segments.append(joined_segment)
            self.__get_all_chant_possibilities(chant_segments = chant_segments,
                                              new_segments = extended_segments,
                                              ith_segment = ith_segment + 1,
                                              epsilon = epsilon,
                                              all_candidates = all_candidates)
        else:
            # don't touch the segment
            self.__dont_touch_segment(chant_segments, new_segments,
                                      ith_segment, epsilon, all_candidates)
            # join the segment with the next one
            self.__join_segments(chant_segments, new_segments,
                                ith_segment, epsilon, all_candidates)
            # split the segment
            self.__split_segment(chant_segments, new_segments,
                                ith_segment, epsilon, all_candidates)


    def __dont_touch_segment(self, chant_segments, new_segments,
                            ith_segment, epsilon, all_candidates):
        # Do not allow melodies that are included in a lot of other chants
        if len(self.segment_inverted_index[chant_segments[ith_segment]]) \
                /self.chant_count < epsilon:
            extended_segments = new_segments.copy()
            extended_segments.append(chant_segments[ith_segment])
            self.__get_all_chant_possibilities(chant_segments = chant_segments,
                                              new_segments = extended_segments,
                                              ith_segment = ith_segment + 1,
                                              epsilon = epsilon,
                                              all_candidates = all_candidates)

    def __join_segments(self, chant_segments, new_segments,
                        ith_segment, epsilon, all_candidates):
        if len(chant_segments) != ith_segment - 1:
            self.__get_all_chant_possibilities(chant_segments = chant_segments,
                                              new_segments = new_segments,
                                              ith_segment = ith_segment + 1,
                                              epsilon = epsilon,
                                              all_candidates = all_candidates,
                                              join_segment = chant_segments[ith_segment])

    def __split_segment(self, chant_segments, new_segments,
                        ith_segment, epsilon, all_candidates):
        # Do not allow melodies that are included in a lot of other chants
        # and do not split one character melody
        if len(self.segment_inverted_index[chant_segments[ith_segment]]) \
                /self.chant_count < epsilon and len(chant_segments[ith_segment]) > 1:


            melody_to_split = chant_segments[ith_segment]

            # Compute size of vocabulary with all potentional new segments
            vocabulary_extension = set()
            for split_point in range(1, len(melody_to_split)):
                vocabulary_extension.add(melody_to_split[:split_point])
                vocabulary_extension.add(melody_to_split[split_point:])

            V_count = len(self.vocabulary) + len(vocabulary_extension)
            for segment in vocabulary_extension:
                if segment in self.vocabulary:
                    V_count = V_count - 1


            # generate all probs of all options
            options = [] # pairs of splited melodies
            probs = [] # probabilities of splited melodies
            for split_point in range(1, len(melody_to_split)):
                left = melody_to_split[:split_point]
                right = melody_to_split[split_point:]
                prob_left = (self.segment_unigrams[left] + 1)/(self.total_segments + V_count)
                prob_right = (self.segment_unigrams[right] + 1)/(self.total_segments + V_count)
                options.append((left, right))
                probs.append(prob_left*prob_right)

            # pick new splited melodies
            probs = np.array(probs)

            option_id = np.random.choice(np.arange(0, len(options)), p=probs/probs.sum())
            left, right = options[option_id]

            # extend segments
            extended_segments = new_segments.copy()
            extended_segments.append(left)
            extended_segments.append(right)
            self.__get_all_chant_possibilities(chant_segments = chant_segments,
                                              new_segments = extended_segments,
                                              ith_segment = ith_segment + 1,
                                              epsilon = epsilon,
                                              all_candidates = all_candidates)


    def __get_candidates_probs(self, all_candidates):
        # Compute size of vocabulary with all potentional new segments
        vocabulary_extension = set()
        for candidate in all_candidates:
            for segment in candidate:
                vocabulary_extension.add(segment)
        V_count = len(self.vocabulary) + len(vocabulary_extension)
        for segment in vocabulary_extension:
            if segment in self.vocabulary:
                V_count = V_count - 1

        # Compute probs for all candidates
        probs = []
        for candidate in all_candidates:
            new_prob = 1
            for segment in candidate:
                new_prob *= (self.segment_unigrams[segment] + 1)/(self.total_segments + V_count)
            probs.append(new_prob)
        probs = np.array(probs)

        return probs/probs.sum()