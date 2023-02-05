import random
import numpy as np
from collections import defaultdict

class ViterbiBasedModel:
    def __init__(self, min_size = 3, max_size = 8, seed = 0):
        random.seed(seed)
        np.random.seed(seed)
        self.min_size = min_size
        self.max_size = max_size
        self.__init_model()

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


    def predict_segments(self, chants, iterations = 5,
                        mu = 5, sigma = 2, print_each = 1):
        self.__init_model()
        self.__generate_vocabulary(chants)
        # Do init segmentation, generate model's dictionaries (segment_unigrams, ...)
        init_segmentation = self.__gaus_rand_segments(chants, mu, sigma)
        # Update data structures
        self.chant_count = len(chants)
        chant_segmentation = init_segmentation
        for i in range(iterations):
            chant_segmentation = self.__train_iteration(chant_segmentation)
            if i%print_each == 0:
                print("{}. Iteration".format(i))
                top25_melodies = sorted(self.segment_unigrams, key=self.segment_unigrams.get, reverse=True)[:30]
                print("\t\t\t", top25_melodies)
                #for topmel in top25_melodies:
                #  print("\t\t\t{}".format(topmel))
        return chant_segmentation

    # ------------------------------- data structures updates -------------------------
    def __generate_vocabulary(self, chants):
        self.vocabulary = set()
        for chant_str in chants:
            self.__update_vocab(chant_str=chant_str, char_id = 0)
        print("Vocabulary was generated..")

    def __update_vocab(self,chant_str: str, char_id: int):
        if not char_id == len(chant_str):
            for segment_boundary_r in range(char_id + self.min_size, 
                                        min((char_id+self.max_size+1), len(chant_str)+1)):
                new_segment = chant_str[char_id:segment_boundary_r]
                self.vocabulary.add(new_segment)
                self.__update_vocab(chant_str=chant_str, char_id=segment_boundary_r)
                
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
                new_len = np.clip(a = max(int(random.gauss(mu, sigma)),
                    a_min = self.min_size, a_max = self.max_size))
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
    def __train_iteration(self, segmented_chants):
        new_segmented_chants = []
        for chant_id, segments in enumerate(segmented_chants):
            self.__ignore_chant(chant_segments = segments, chant_id = chant_id)
            new_segments = self.__get_optimized_chant(chant_segments = segments)
            self.__add_chant(chant_segments = new_segments, chant_id = chant_id)
            new_segmented_chants.append(new_segments)
        return new_segmented_chants


    def __get_optimized_chant(self, chant_segments, k_best = 15):
        chant = ''.join(chant_segments)
        trellis = [[]]*(len(chant)+1) # for each melody pitch, store the list of k_best nodes (prob, position, prev_node)
        trellis[0] = [Node(0, 1, None)] # position = 0, prob = 1, prev_node = None
        self.__chant_viterbi_traversal(chant_str=chant, chant_id = 0, 
                                        trellis=trellis, k_best=k_best)
        return self.__decode_trellis(trellis)



    def __chant_viterbi_traversal(self, chant_str: str, char_id:int, trellis, k_best: int):
        if not char_id == len(chant_str):
            for segment_boundary_r in range(char_id + self.min_size, 
                                        min((char_id+self.max_size+1), len(chant_str)+1)):
                # update trellis
                new_segment = chant_str[char_id:segment_boundary_r]
                self.__update_trellis(trellis, segment_boundary_r, new_segment, k_best)
                # recursive call
                self.__chant_viterbi_traversal(chant_str, segment_boundary_r, trellis, k_best)



    def __decode_trellis(self, graph: list, chant_str: str):
        final_segmentation_ids = []
        # Choose the final path
        probs = np.array([path.prob for path in graph[-1]])
        final_node = np.random.choice(np.arange(0, len(probs)), p=probs/probs.sum())
        final_segmentation_ids.append(final_node.position)
        # find segment ids
        while final_node.prev_node != None:
            final_node = final_node.prev_node
            final_segmentation_ids.append(final_node.position)

        # Final segmentation ids into segmentations
        final_segmentation = []
        for i, j in zip(final_segmentation_ids[:-1], final_segmentation_ids[1:]):
            final_segmentation.append(chant_str[j:i])

        return final_segmentation.reverse()



    def __update_trellis(self, graph, id:int, new_segment:str, V:int, k_best:int):
        prev_id = id - len(new_segment)
        V = len(self.vocabulary)
        new_segment_prob = (self.segment_unigrams[new_segment] + 1)/(self.total_segments + V)
        potential_candidates = graph[id]
        for node in graph[prev_id]:
            new_prob = node.prob*new_segment_prob
            potential_candidates.append(Node(id, new_prob, node))

        potential_candidates = sorted(potential_candidates, key=lambda x: x.prob)

        graph[id] = potential_candidates[:k_best]

class Node:
    def __init__(self, position:int, prob:float, prev_node: "Node"):
        self.position = position
        self.prob = prob
        self.prev_node = prev_node