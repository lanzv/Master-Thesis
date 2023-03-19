from src.models.nhpylm.npylm import NPYLM
import numpy as np
from src.models.nhpylm.chant import Chant
from src.models.nhpylm.definitions import BOS, EOS, BOW, EOW
"""
This structs holds all the necessary fields for sampling chant segmentations using forward-backward inference.
"""
class Sampler:
    def __init__(self, npylm: "NPYLM", min_word_length: int, max_word_length: int, max_chant_length: int):
        self.npylm: "NPYLM" = npylm
        # The word_ids of the current 3-gram being calculated
        self.word_ids: list = [0 for _ in range(3)]
        # Cache of ids of some words previously segmented in this chant.
        # substring_word_id_cache::Array{Int, 2}
        self.substring_word_id_cache: list = np.zeros((max_chant_length+1, max_word_length+1))
        #3-dimensional tensor that contains the forward variables, i.e. in α[t][k][j] at p.104 of the paper
        # α_tensor::Array{Float64, 3}
        self.alpha_tensor: list = np.zeros((max_chant_length+1, max_word_length+1, max_word_length+1)) # OffsetArray{Float64}

        # Cache of word probabilities p_w given by the CHPYLM. Caching the value is useful so that we can avoid excessive repeated computations.
        # I think the "h" here actually stands for Theta or something
        # Indexing: p_w_h_cache[t][k][j][i]
        # - t: The full length of the chant
        # - k: The length of the third gram, the last word
        # - j: The length of the second gram
        # - i: The length of the first gram
        # One problem is that we can't use zero indexing. So in cases where the first gram is BOS we still need to do something else.
        # Still, as I suggested, why don't we just make BOS inherently a part of the chant when we read it in. Is there any problem with that?
        # p_w_h_cache::Array{Float64, 4}
        self.p_w_h_cache: list = np.zeros((max_chant_length+1, max_word_length+1, max_word_length+1, max_word_length+1)) # OffsetArray{Float64}
        # Normalization constants
        self.log_z: list = [0.0 for _ in range(max_chant_length + 1)] # OffsetVector{Float64}
        # Stores the inverse of the actual probabilities (one for each length). Used when `with_scaling`==true. The purpose is to combat probabilities that are way too low.
        self.scaling_coefficients: list = [0.0 for _ in range(max_chant_length + 2)] # OffsetVector{Float64}
        # Table to temporarily hold possibilities for the sampling of j and k during the backward sampling.
        # It's called a "table" because we first record the probabilities for all candidates j and k values. After that, we need to *draw* from this "table" actual j and k values.
        # See line 8 of Figure 5 of the paper.
        self.backward_sampling_table: list = [0.0 for _ in range(max_word_length * max_word_length)] # OffsetVector{Float64}
        # Matrix to hold the indices of i that maximize the log probability of the trigram sequence.
        # e.g. viterbi_backward_indices[t,k,j] = 2 means when the first gram (i) has length 2, the probability is maximized. This is why there isn't a `i` index, unlike the p_w_h_cache array
        self.viterbi_backward_indices: list = np.zeros((max_chant_length+1, max_word_length+1, max_word_length+1)) # OffsetArray{Int, 3}

        # I can probably make this one non-mutable if I change the representation of these two a bit. Let's see.
        # This is L in the paper, i.e. the maximum length allowed for a word.
        self.max_word_length: int = max_word_length
        self.min_word_length: int = min_word_length
        # I feel that this is just a convenience variable to preallocate array space so that even the longest chant can be accomodated. I don't think the model itself has any sort of max length restriction on chants.
        self.max_chant_length: int = max_chant_length

    def extend_capacity(self, max_word_length: int, max_chant_length: int):
        # If the existing capacity is already good enough then no need to extend
        if (max_word_length <= self.max_word_length and max_chant_length <= self.max_chant_length):
            return
        else:
            self.allocate_capacity(max_word_length, max_chant_length)

    def allocate_capacity(self, max_word_length: int, max_chant_length: int):
        self.max_word_length = max_word_length
        self.max_chant_length = max_chant_length
        # Size of arrays that contain the various values. Because we need to take care of situations involving BOS, the size of such arrays should be 1 longer than only the max_chant_length
        size = max_chant_length + 1
        self.log_z = [0.0 for _ in range(max_chant_length + 1)]
        # Why does this need to be one longer?
        self.scaling_coefficients = [0.0 for _ in range(size + 1)]
        self.viterbi_backward_indices = np.zeros((max_chant_length + 1, max_word_length + 1, max_word_length + 1))
        self.backward_sampling_table = [0.0 for _ in range(max_word_length* max_word_length)]

        # It needs to be 1 longer than length, because we have to accomodate for the index 0, which indicates that we have BOS as one of the grams.
        self.alpha_tensor = np.zeros((size+1, max_word_length+1, max_word_length+1))
        # sampler.p_w_h_cache = Array{Float64, 4}(undef, size, max_word_length + 1, max_word_length + 1, max_word_length + 1)
        self.p_w_h_cache = np.zeros((max_chant_length+1, max_word_length+1, max_word_length+1, max_word_length+1))
        # sampler.substring_word_id_cache = Array{Int, 2}(undef, size, max_word_length + 1)
        self.substring_word_id_cache = np.zeros((max_chant_length+1, max_word_length+1))

    def get_substring_word_id_at_t_k(self, chant: "Chant", t: int, k: int):
        """
        α[t][k][j] represents the marginal probability of string c1...ct with both the final k characters and further j preceding characters being words.
        This function returns the id of the word constituted by the last k characters of the total t characters.
        Note that since this function already takes care of the index shift that's needed in Julia, the callers will still just call it normally.
        """
        word_id = self.substring_word_id_cache[t,k]
        # 0 is used to indicate the initial state, where there's no cache.
        # Though wouldn't it conflict with BOS? Let's see then.
        if word_id == 0:
            # Fuck the Julia indexing system. Let me stick to 0-based indexing for now.
            word_id = chant.get_substr_word_id(t - k, t - 1)
            self.substring_word_id_cache[t,k] = word_id

        return word_id

    def forward_filtering(self, chant: "Chant", with_scaling: bool):
        """
        Performs the forward filtering on the target chant.
        """
        self.alpha_tensor[0,0,0] = 1.0
        for t in range(1, chant.length() + 1):
            prod_scaling = 1.0
            # The original paper most likely made a mistake on this. Apparently one should take min(t, L) instead of max(1, t - L) which makes no sense.
            for k in range(1, min(t, self.max_word_length) + 1):
                if (with_scaling and k > 1):
                    # TODO: Why + 1 though. Need to understand prod_scaling better.
                    prod_scaling *= self.scaling_coefficients[t - k + 1]
                # If t - k = 0 then the loop will not be executed at all.
                for j in range(0 if t == k else 1,  min(t-k, self.max_word_length) + 1):
                    self.alpha_tensor[t,k,j] = 0
                    self.calculate_alpha_t_k_j(chant, t, k, j, prod_scaling)
            # Perform scaling operations on the alpha tensor, in order to avoid underflowing.
            if (with_scaling):
                sum_alpha = 0.0
                for k in range(1, min(t, self.max_word_length)+1):
                    for j in range(0 if t == k else 1, min(t - k, self.max_word_length) + 1):
                        sum_alpha += self.alpha_tensor[t,k,j]
                if not sum_alpha > 0.0:
                    raise Exception("sum_alpha <= 0.0")
                self.scaling_coefficients[t] = 1.0 / sum_alpha
                for k in range(1, min(t, self.max_word_length) + 1):
                    for j in range(0 if t == k else 1, min(t - k, self.max_word_length) + 1):
                        self.alpha_tensor[t,k,j] *= self.scaling_coefficients[t]

    def calculate_alpha_t_k_j(self, chant: "Chant", t: int, k: int, j: int, prod_scaling: float):
        # If α[t-k][j][i] is already normalized, there's no need to normalize α[t][k][j]
        """
        The step during forward filtering where α[t][k][j] is calculated
        Note that in this trigram case, α[t][k][j] = \sum^{t-k-j}_{i=1} p(c^t_{t-k+1} | c^{t-k-j}_{t-k-j-i+1} c^{t-k}_{t-k-j+1}) * α[t - k][j][i]
        That is to say, we first fix both the third gram and the second gram, and marginalize over different lengths of the first gram, indicated by the changing index i here.
        """
        word_k_id = self.get_substring_word_id_at_t_k(chant, t, k)
        chant_as_chars = chant.characters
        if not (t <= self.max_chant_length + 1):
            raise Exception("(t > self.max_chant_length + 1)")
        if not (k <= self.max_word_length):
            raise Exception("k > self.max_word_length")
        if not (j <= self.max_word_length):
            raise Exception("j > self.max_word_length")
        if not (t - k >= 0):
            raise Exception("t - k <= 0")
        # I'm really unsatisfied with the constant manual generation of BOS and EOS. I mean why not generate it already when first reading in the corpus? This can probably save tons of problems.
        # However, I can now also see why this might be necessary: i.e. so that BOS and EOS, which are essentially special characters, might not be accidentally considered a part of a word when we try to determine word boundaries, which would result in nonsensical results... Um let's see. Let me first port the code anyways.

        # Speical case 1: j == 0 means there's no actual *second* gram, i.e. the first two tokens are both BOS!
        if j == 0:
            self.word_ids[0] = BOS
            self.word_ids[1] = BOS
            self.word_ids[2] = word_k_id
            # Compute the probability of this word with length k
            # Why do we - 1 in the end though?
            p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t - k, t - 1)
            if not p_w_h > 0.0:
                raise Exception("p_w_h <= 0.0")
            # I think the scaling is to make sure that this thing doesn't underflow.
            # Store the values in the cache
            self.alpha_tensor[t,k,0] = p_w_h * prod_scaling
            self.p_w_h_cache[t,k,0,0] = p_w_h
            return
        # Special case 2: This is the case where i == 0 but j != 0, i.e. the first gram is BOS (but the second gram is a normal word)
        elif t - k - j == 0:
            word_j_id = self.get_substring_word_id_at_t_k(chant, t - k, j)
            self.word_ids[0] = BOS
            self.word_ids[1] = word_j_id
            self.word_ids[2] = word_k_id
            # Probably of the word with length k, which is the last (3rd) word.
            p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t - k, t - 1)
            if not p_w_h > 0.0:
                raise Exception("p_w_h <= 0.0")
            if not self.alpha_tensor[t-k,j,0] > 0.0:
                raise Exception("self.alpha_tensor[t-k,j,0] <= 0.0")
            # In this case, the expression becomes the following.
            self.alpha_tensor[t,k,j] = p_w_h * self.alpha_tensor[t - k,j,0] * prod_scaling
            # The last index here is i. i == 0.
            self.p_w_h_cache[t,k,j,0] = p_w_h
            return
        else:
            # Perform the normal marginalization procedure in all other cases
            sum = 0.0
            for i in range(1, min(t - k - j, self.max_word_length)+1):
                word_i_id = self.get_substring_word_id_at_t_k(chant, t - k - j, i)
                word_j_id = self.get_substring_word_id_at_t_k(chant, t - k, j)
                # The first gram
                self.word_ids[0] = word_i_id
                # The second gram
                self.word_ids[1] = word_j_id
                # The third gram
                self.word_ids[2] = word_k_id

                # This way of writing the code is still a bit messy. Let's see if we can do better then.
                p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t - k, t - 1)
                if not p_w_h > 0.0:
                    raise Exception("p_w_h <= 0.0")
                if not i <= self.max_word_length:
                    raise Exception("i > self.max_word_length")
                if not self.alpha_tensor[t-k,j,i] > 0.0:
                    raise Exception("self.alpha_tensor[t-k,j,i]")
                # Store the word possibility in the cache tensor.
                self.p_w_h_cache[t,k,j,i] = p_w_h
                temp = p_w_h * self.alpha_tensor[t - k,j,i]
                sum += temp
            if not sum > 0.0:
                raise Exception("num <= 0.0")
            self.alpha_tensor[t,k,j] = sum * prod_scaling

    def backward_sampling(self, chant: "Chant"):
        """
        Performs the backward sampling on the target chant.
        """
        t = chant.length()
        k = [0]
        j = [0]
        sum_length = 0
        self.backward_sample_k_and_j(chant, t, 1, k, j)

        # I'm not sure yet why the segments array should contain their lengths instead of the word ids themselves. I guess it's just a way to increase efficiency and all that.
        segment_lengths = []

        # Record the last word we just sampled.
        segment_lengths.append(k[0])

        # Deal with the special case: There's only one word in total for the chant.
        if j[0] == 0 and k[0] == t:
            segment_lengths.reverse()
            return segment_lengths

        if not (k[0] > 0 and j[0] > 0):
            raise Exception("k[0] <= 0 or j[0] <= 0")
        if not (j[0] <= self.max_word_length):
            raise Exception("j[0] > self.max_word_length")

        # Record the second-to-last word we just sampled.
        segment_lengths.append(j[0])
        t -= (k[0] + j[0])
        sum_length += k[0] + j[0]
        next_word_length = j[0]

        while (t > 0):
            # There's only ever one character left in the whole chant
            if t == 1:
                k[0] = 1
                j[0] = 0
            else:
                self.backward_sample_k_and_j(chant, t, next_word_length, k, j)
            segment_lengths.append(k[0])
            t -= k[0]
            if j[0] == 0:
                # println("t is $(t)")
                if not (t == 0):
                    raise Exception("t != 0")
            else:
                if not (j[0] <= self.max_word_length):
                    raise Exception("j[0] > self.max_word_length")
                segment_lengths.append(j[0])
                t -= j[0]
            sum_length += (k[0] + j[0])
            next_word_length = j[0]
        if not (t == 0):
            raise Exception("t != 0")
        if not (sum_length == chant.length()):
            raise Exception("sum_length != chant.length()")
        # result = OffsetArray(reverse(segment_lengths), 0:length(segment_lengths) - 1)
        # println("In backward_sampling, chant is $chant, segment_lengths is $segment_lengths, result is $result")
        segment_lengths.reverse()
        return segment_lengths

    def backward_sample_k_and_j(self, chant: "Chant", t: int, third_gram_length: int, sampled_k: list, sampled_j: list):
        """
        Returns k and j in a tuple, which denote the offsets for word boundaries of the two words we are interested in sampling.
        "next_word" really means the target word, the last gram in the 3 gram, e.g. the EOS in p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
        """
        table_index = 0
        chant_as_chars = chant.characters
        chant_length = chant.length()
        sum_p = 0.0
        for k in range(1, min(t, self.max_word_length) + 1):
            for j in range(1, min(t - k, self.max_word_length)+1):
                word_j_id = self.get_substring_word_id_at_t_k(chant, t - k, j)
                word_k_id = self.get_substring_word_id_at_t_k(chant, t, k)
                # When we begin the backward sampling on a chant, note that the final token is always EOS. We have probabilities p(EOS | c^N_{N - k + 1} c^{N-k}_{N-k-j + 1}) * α[N][k][j])
                word_t_id = EOS
                if t < chant.length():
                    if not (t + third_gram_length <= chant.length()):
                        raise Exception("t + third_gram_length >= chant.length()")
                    if not (third_gram_length > 0):
                        raise Exception("third_gram_length <= 0")
                    # Otherwise the final token is not EOS already but an actual word. Still the principles for sampling don't change.
                    word_t_id = self.get_substring_word_id_at_t_k(chant, t + third_gram_length, third_gram_length)
                self.word_ids[0] = word_j_id
                self.word_ids[1] = word_k_id
                self.word_ids[2] = word_t_id
                p_w_h = 0.0
                if t == chant.length():
                    # The only exception to caching is the situation where the last gram is EOS.
                    # p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
                    p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t, t)
                else:
                    # In all other scenarios, we should have already cached this value.
                    p_w_h = self.p_w_h_cache[t + third_gram_length, third_gram_length, k, j]
                if not (self.alpha_tensor[t,k,j] > 0):
                    raise Exception("self.alpha_tensor[t, k, j] <= 0")
                # p(3rd_gram | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
                p = p_w_h * self.alpha_tensor[t,k,j]
                if not (p > 0):
                    raise Exception("p <= 0")
                self.backward_sampling_table[table_index] = p
                sum_p += p
                # println("p_w_h is $(p_w_h), sampler.α_tensor[t,k,j] is $(sampler.α_tensor[t,k,j]), p is $(p), sum_p is $(sum_p)")
                table_index += 1
            # In this case the first gram is BOS. The third gram is EOS.
            # This is a kind of a special case that needs to be taken care of, since if t - k == 0, the inner loop will be `for j in 1:0`, i.e. it will never be executed at all.
            # TODO: One can likely refactor this code a bit more. Just setting j = 0 should be enough shouldn't it?
            if t == k:
                j = 0
                word_j_id = BOS
                word_k_id = self.get_substring_word_id_at_t_k(chant, t, k)
                word_t_id = EOS
                if t < chant.length():
                    if not (t + third_gram_length <= chant.length()):
                        raise Exception("t + third_gram_length > chant.length()")
                    if not (third_gram_length > 0):
                        raise Exception("third_gram_length <= 0")
                    word_t_id = self.get_substring_word_id_at_t_k(chant, t + third_gram_length, third_gram_length)
                self.word_ids[0] = word_j_id
                self.word_ids[1] = word_k_id
                self.word_ids[2] = word_t_id
                p_w_h = 0.0
                if t == chant.length():
                    # p(EOS | c^N_{N-k} c^{N-k}_{N-k-j})
                    p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t, t)
                else:
                    # We should have already cached this value.
                    p_w_h = self.p_w_h_cache[t + third_gram_length, third_gram_length, k, j]
                if not (self.alpha_tensor[t,k,j] > 0):
                    raise Exception("self.alpha_tensor[t, k, j] <= 0")
                # p(3rd_gram | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
                p = p_w_h * self.alpha_tensor[t,k,j]
                if not (p > 0):
                    raise Exception("p <= 0")
                self.backward_sampling_table[table_index] = p
                sum_p += p
                table_index += 1


        if not (table_index > 0):
            raise Exception("table_index <= 0")
        if not (table_index <= self.max_word_length * self.max_word_length):
            raise Exception("table_index > self.max_word_length * self.max_word_length")

        # Eventually, the table should have (min(t, sampler.max_word_length) * min(t - k, sampler.max_word_length)) + 1 entries
        # This is such a pain. We should definitely be able to simplify the code much more than this. Eh.
        normalizer = 1.0 / sum_p
        # println("Normalizer is $(normalizer), sum_p is $(sum_p)")
        randnum = np.random.uniform(0, 1)
        index = 0
        stack = 0.0
        for k in range(1, min(t, self.max_word_length)+1):
            for j in range(1, min(t - k, self.max_word_length)+1):
                if not (index < table_index):
                    raise Exception("index >= table_index")
                if not (self.backward_sampling_table[index] > 0.0):
                    raise Exception("self.backward_sampling_table[index] <= 0.0")
                # Each unique index corresponds to a unique [k, j] combination.
                stack += self.backward_sampling_table[index] * normalizer
                # println("randnum is $(randnum), stack is $(stack)")
                if randnum < stack:
                    sampled_k[0] = k
                    sampled_j[0] = j
                    return
                index += 1

            # The special case where the first gram is BOS. The last entry of the table.
            if t == k:
                # println("t == k triggered!")
                if not (index < table_index):
                    raise Exception("index >= table_index")
                if not (self.backward_sampling_table[index] > 0.0):
                    raise Exception("self.backward_sampling_table[index] <= 0.0")
                stack += self.backward_sampling_table[index] * normalizer
                if randnum < stack:
                    sampled_k[0] = k
                    sampled_j[0] = 0
                    return
                index += 1
        # Sometimes this can somehow fall through?
        print("Fell through!")

    def blocked_gibbs_segment(self, chant: "Chant", with_scaling: bool):
        """
        Does the segment part in the blocked Gibbs algorithm (line 6 of Figure 3 of the paper)
        """
        for i in range(0, chant.length() + 1):
            for j in range(0, self.max_word_length + 1):
                self.substring_word_id_cache[i,j] = 0

        self.forward_filtering(chant, with_scaling)
        return self.backward_sampling(chant)

    def viterbi_argmax_calculate_alpha_t_k_j(self, chant: "Chant", t: int, k: int, j:int):
        # For the 3-gram case, we need to use viterbi decoding to eventually produce the most likely sequence of segments.
        # TODO: OK so what's the difference between the viterbi methods and the original methods without viterbi? I'm kinda lost again. Guess I'll first have to look through the whole training and evaluation flows to look for clues then. Let's see.
        word_k_id = self.get_substring_word_id_at_t_k(chant, t, k)
        chant_as_chars = chant.characters
        if not (t <= self.max_chant_length + 1):
            raise Exception("t > self.max_chant_length+1")
        if not (k <= self.max_word_length):
            raise Exception("k > self.max_word_length")
        if not (j <= self.max_word_length):
            raise Exception("j > self.max_word_length")
        # Special case 1: j == 0 means there's no actual *second* gram, i.e. the first two tokens are both BOS!
        if j == 0:
            self.word_ids[0] = BOS
            self.word_ids[1] = BOS
            self.word_ids[2] = word_k_id
            # Compute the probability of this word with length k
            # Why do we - 1 in the end though? We probably shouldn't do so here since the indexing system is different. Eh.
            p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t - k, t - 1)
            if not (p_w_h > 0.0):
                raise Exception("p_w_h <= 0.0")
            # I think the scaling is to make sure that this thing doesn't underflow.
            # Store the values in the cache
            self.alpha_tensor[t,k,0] = np.log(p_w_h)
            # Here is the difference.
            self.viterbi_backward_indices[t,k,0] = 0
            return
        # Special case 2: This is the case where i == 0 but j != 0, i.e. the first gram is BOS (but the second gram is a normal word)
        elif t - k - j == 0:
            word_j_id = self.get_substring_word_id_at_t_k(chant, t - k, j)
            self.word_ids[0] = BOS
            self.word_ids[1] = word_j_id
            self.word_ids[2] = word_k_id
            # Probability of the word with length k, which is the last (3rd) word.
            p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t - k, t - 1)
            if not (p_w_h > 0.0):
                raise Exception("p_w_h <= 0.0")
            if not (self.alpha_tensor[t-k,j,0] != 0.0):
                raise Exception("self.alpha_tensor[t-k, j, 0] = 0.0")
            # In this case, the expression becomes the following.
            self.alpha_tensor[t,k,j] = np.log(p_w_h) + self.alpha_tensor[t - k,j,0]
            self.viterbi_backward_indices[t,k,j] = 0
            return
        else:
            # Perform the normal marginalization procedure in all other cases
            max_log_p = 0.0
            argmax = 0
            for i in range(1, min(t - k - j, self.max_word_length) + 1):
                word_i_id = self.get_substring_word_id_at_t_k(chant, t - k - j, i)
                word_j_id = self.get_substring_word_id_at_t_k(chant, t - k, j)
                # The first gram
                self.word_ids[0] = word_i_id
                # The second gram
                self.word_ids[1] = word_j_id
                # The third gram
                self.word_ids[2] = word_k_id

                p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t - k, t - 1)
                if not (p_w_h > 0.0):
                    raise Exception("p_w_h <= 0")
                if not (i <= self.max_word_length):
                    raise Exception("i > self.max_word_length")
                # Because it's a log value then.
                if not (self.alpha_tensor[t-k,j,i] <= 0):
                    raise Exception("self.alpha_tensor[t-k, j, i] > 0")
                temp = np.log(p_w_h) + self.alpha_tensor[t - k,j,i]
                if not (temp <= 0):
                    raise Exception("temp > 0")

                # We're trying to determine the i value (first gram) that maximizes the possibility
                if (argmax == 0 or temp > max_log_p):
                    argmax = i
                    max_log_p = temp
            if not (argmax > 0):
                raise Exception("argmax <= 0")
            self.alpha_tensor[t,k,j] = max_log_p
            # We use the viterbi_backward_indices matrix to store the i value that maximizes the possibility of the trigram.
            self.viterbi_backward_indices[t,k,j] = argmax


    def viterbi_forward_filtering(self, chant: "Chant"):
        for t in range(1, chant.length() + 1):
            for k in range(1, min(t, self.max_word_length) + 1):
                # There is no j, i.e. the second gram is also BOS.
                if t == k:
                    self.viterbi_argmax_calculate_alpha_t_k_j(chant, t, k, 0)
                # Note that in the t==k case, we will have range 1:0 which is automatically empty, so the following code will not be run.
                for j in range(1, min(t - k, self.max_word_length)+1):
                    self.viterbi_argmax_calculate_alpha_t_k_j(chant, t, k, j)

    def viterbi_argmax_backward_sample_k_and_j_to_eos(self, chant: "Chant", t:int, third_gram_length: int, argmax_k: list, argmax_j: list):
        """
        This method is called when we know the third gram is EOS, so we're only sampling the first gram and second gram.
        """
        if not (t == chant.length()):
            raise Exception("t != chant.length()")
        table_index = 0
        chant_as_chars = chant.characters
        chant_length = chant.length()
        max_log_p = 0.0
        argmax_k[0] = 0
        argmax_j[0] = 0
        for k in range(1, min(t, self.max_word_length) + 1):
            for j in range(1, min(t - k, self.max_word_length) + 1):
                word_j_id = self.get_substring_word_id_at_t_k(chant, t - k, j)
                word_k_id = self.get_substring_word_id_at_t_k(chant, t, k)
                # When we begin the backward sampling on a chant, note that the final token is always EOS. We have probabilities p(EOS | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
                self.word_ids[0] = word_j_id
                self.word_ids[1] = word_k_id
                self.word_ids[2] = EOS
                # It's still the EOS. We just wrote it in a simpler way.
                p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t, t)
                if not self.alpha_tensor[t,k,j] <= 0:
                    raise Exception("self.alpha_tensor[t,k,j] > 0")
                temp = np.log(p_w_h) + self.alpha_tensor[t,k,j]
                if not temp <= 0:
                    raise Exception("temp > 0")
                if (argmax_k[0] == 0 or temp > max_log_p):
                    max_log_p = temp
                    argmax_k[0] = k
                    argmax_j[0] = j
            # TODO: One can likely refactor this code a bit more.
            # In this case the first gram is BOS. The third gram is EOS.
            if t == k:
                word_j_id = BOS
                word_k_id = self.get_substring_word_id_at_t_k(chant, t, k)
                word_t_id = EOS
                # We removed all code regarding cases where t < length of the original chant. I think this is because we always know that this will be the case where t == length(chant) and that the third gram will always be the EOS, i.e. this method will only be called in such cases.
                # So apparently we can simplify the code a bit and maybe put it together with another method eh?
                self.word_ids[0] = word_j_id
                self.word_ids[1] = word_k_id
                self.word_ids[2] = word_t_id
                p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t, t)
                if not self.alpha_tensor[t,k,0] <= 0:
                    raise Exception("self.alpja_tensor[t,k,0] > 0")
                # p(3rd_gram | c^N_{N-k} c^{N-k}_{N-k-j}) * α[N][k][j])
                temp = np.log(p_w_h) + self.alpha_tensor[t,k,0]
                if not temp <= 0:
                    raise Exception("temp > 0")
                if argmax_k[0] == 0 or temp > max_log_p:
                    max_log_p = temp
                    argmax_k[0] = k
                    argmax_j[0] = 0

    def viterbi_backward_sampling(self, chant: "Chant"):
        segment_lengths = []
        t = chant.length()
        sum_length = 0
        kc = [0]
        jc = [0]
        self.viterbi_argmax_backward_sample_k_and_j_to_eos(chant, t, 1, kc, jc)
        k = kc[0]
        j = jc[0]
        if not (k <= self.max_word_length):
            raise Exception("k > self.max_word_length")
        # I'm not sure yet why the segments array should contain their lengths instead of the word ids themselves. I guess it's just a way to increase efficiency and all that.
        segment_lengths.append(k)
        sum_length += k

        # There's only one word in total for the chant.
        if j == 0 and k == t:
            segment_lengths.reverse()
            return segment_lengths


        if not (k > 0 and j > 0):
            raise Exception("k <= 0 or j <= 0")
        if not (j <= self.max_word_length):
            raise Exception("j > self.max_word_length")
        segment_lengths.append(j)
        # We knew that i is the index that maximizes the possibility of the trigram
        i = self.viterbi_backward_indices[int(t),int(k),int(j)]
        if not i >= 0:
            raise Exception("j < 0")
        if not i <= self.max_word_length:
            raise Exception("i > self.max_word_length")
        sum_length += j + i

        # Move the "chant end" forward
        t -= k
        k = j
        j = i

        # The chant is already fully segmented.
        if i == 0:
            if not sum_length == chant.length():
                raise Exception("sum_length != chant.length()")
            segment_lengths.reverse()
            return segment_lengths

        segment_lengths.append(i)

        # TODO: This code repetition surely is also avoidable?
        # Repeatedly push forward the end, taking advantage of the already recorded viterbi indices.
        while (t > 0):
            i = self.viterbi_backward_indices[int(t),int(k),int(j)]
            if not i >= 0:
                raise Exception("i < 0")
            if not i <= self.max_word_length:
                raise Exception("i > self.max_word_length")
            if (i != 0):
                segment_lengths.append(i)
            t -= k
            k = j
            j = i
            sum_length += i
        if not (t == 0):
            raise Exception("t != 0")
        if not (sum_length == chant.length()):
            raise Exception("sum_length != chant.length()")
        if not chant.length() > 0:
            raise Exception("chant.length() <= 0")
        segment_lengths.reverse()
        return segment_lengths

    def viterbi_decode(self, chant: "Chant"):
        """
        Viterbi decoding algorithm is used to find the best segmentation of a chant, given an **already learned** model. This method is used when we want to perform evaluation on test data.
        """
        # array_length = length(chant) + 1
        self.alpha_tensor[0,0,0] = 0.0
        self.log_z[0] = 0.0
        for t in range(0, chant.length()+1):
            for k in range(0, self.max_word_length+1):
                self.substring_word_id_cache[t,k] = 0
        self.viterbi_forward_filtering(chant)
        return self.viterbi_backward_sampling(chant)

    def compute_log_forward_probability(self, chant: "Chant", with_scaling: bool):
        """
        Computes the probability of resulting in EOS with the given α_tensor for the chant.
        """
        self.enumerate_forward_variables(chant, with_scaling)
        # It should now point to EOS
        t = chant.length() + 1
        if not with_scaling:
            # Consider the length of EOS to be 1
            k = 1
            alpha_eos = 0.0
            # As described in the paper, we need to sum all possible previous permutations together in this case.
            for j in range(1, min(t - k, self.max_word_length)+1):
                if not (self.alpha_tensor[t,k,j] > 0.0):
                    raise Exception("self.alpha_tensor[t,k,j] <= 0.0")
                alpha_eos += self.alpha_tensor[t,k,j]
            if not (alpha_eos > 0.0):
                raise Exception("alpha_eos <= 0.0")
            return np.log(alpha_eos)
        else:
            # If we use scaling, we stored the scaling coefficients as the inverse of the actual probabilities.
            log_p_x = 0.0
            for i in range(1, t+1):
                log_p_x += np.log(1.0 / self.scaling_coefficients[i])
            return log_p_x

    def enumerate_forward_variables(self, chant: "Chant", with_scaling: bool):
        # array_length = length(chant) + 1
        for i in range(0, chant.length() + 1):
            for j in range(0, self.max_word_length + 1):
                self.substring_word_id_cache[i,j] = 0

        chant_as_chars = chant.characters
        # This should fill the alpha tensor before thef final EOS.
        self.forward_filtering(chant, with_scaling)
        # Calculate the possibility of producing EOS as the final gram.
        # Though isn't EOS also a part of the original chant? Doesn't seem to be the case then due to the special processing I guess. Still a bit confusing that's for sure... Let's just go on and do better then. Eh.
        alpha_eos = 0.0
        t = chant.length() + 1
        k = 1
        for j in range(1, min(t - k, self.max_word_length) + 1):
            prob_sum = 0.0
            for i in range(0 if (t - k - j == 0) else 1, min(t - k - j, self.max_word_length) + 1):
                self.word_ids[0] = self.get_substring_word_id_at_t_k(chant, t - k - j, i)
                self.word_ids[1] = self.get_substring_word_id_at_t_k(chant, t - k, j)
                self.word_ids[2] = EOS
                p_w_h = self.npylm.compute_p_w_of_nth_word_chars(chant_as_chars, self.word_ids, 2, t, t)
                if not (p_w_h > 0.0):
                    raise Exception("p_w_h <= 0.0")
                prob_sum += p_w_h * self.alpha_tensor[t-k,j,i]
            self.alpha_tensor[t,k,j] = prob_sum
            alpha_eos += prob_sum
        if with_scaling:
            self.scaling_coefficients[t] = 1.0 / alpha_eos