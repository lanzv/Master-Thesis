from src.models.nhpylm.whpylm import WHPYLM
from src.models.nhpylm.chpylm import CHPYLM
from src.models.nhpylm.pyp import PYP
from src.models.nhpylm.definitions import BOW, EOW, BOS, EOS, EOS_CHAR
from src.models.nhpylm.wtype import WORDTYPE_NUM_TYPES, detect_word_type_substr
import numpy as np
from scipy.stats import poisson
from src.models.nhpylm.chant import Chant

class NPYLM:

    def __init__(self, min_word_length: int, max_word_length: int, max_chant_length: int, G_0: float, initial_lambda_a: float, initial_lambda_b: float, chpylm_beta_stop: float, chpylm_beta_pass: float, n_gram: int):
        self.n_gram = n_gram
        # The hierarhical Pitman-Yor model for words
        self.whpylm: "WHPYLM" = WHPYLM(n_gram) # 2 for bigram, 3 for trigram, others are not supported
        # The hierarhical Pitman-Yor model for characters
        self.chpylm: "CHPYLM" = CHPYLM(G_0, max_chant_length, chpylm_beta_stop, chpylm_beta_pass)


        # Each key represents a word.
        # Remember that we feed in a word like a "chant", i.e. run `add_customer` char-by-char, into the CHPYLM.
        # We need to record the depth of each char when it was first added, so that we can remove them at the same depths later, when we remove the word from the CHPYLM.
        # Remember that each dish can be served at multiple tables, i.e. there is a certain probability that a customer sits at a new table.
        # Therefore, the outermost Vector in Vector{Vector{Int}} keeps tracks of the different tables that this token is served at!
        # This is useful so that when we later remove
        # Compare it with the field `tablegroups::Dict{T, Vector{Int}}` in PYP.jl
        # For the *innermost* Vector{Int}, the index of the entry corresponds to the char index, the value of the entry corresponds to the depth of that particular char entry.
        # This is to say, for *a particular table* that the token is served at, the breakdown of char depths is recorded in that vector.
        # Normal `Vector` is used so that the first char of the word is accessed via index 1. Probably not a good idea? I guess it still works as long as the whole system is consistent. Let's see.
        # Yeah I think it might be a better idea to keep these things always constant anyways. Let's see.
        self.recorded_depth_arrays_for_tablegroups_of_token = {} #Dict{UInt, Vector{OffsetVector{Int}}}

        # The cache of WHPYLM G_0. This will be invalidated once the seating arrangements in the CHPYLM change.
        self.whpylm_G_0_cache = {} # Dict{UInt, Float64}
        # The cache of CHPYLM G_0
        self.chpylm_G_0_cache = {} # Dict{Int, Float64}
        self.lambda_for_types = [0.0 for _ in range(WORDTYPE_NUM_TYPES + 1)] # OffsetVector{Float64}
        # Probability of generating a word of length k from the CHPYLM
        self.p_k_chpylm = [1.0 / (max_word_length + 2) for _ in range(max_word_length + 2)] # .. trigram .. OffsetVector{Float64}
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length # Int
        self.max_chant_length = max_chant_length #Int
        # The shape parameter of the Gamma distribution for estimating the λ value for the Poisson distribution. (Expression (16) of the paper)
        # The relation: λ ~ Ga(a, b), where a is the shape and b is rate (not scale)
        self.lambda_a = initial_lambda_a # Float64
        # The rate (not scale) parameter of the Gamma distribution for estimating the λ value for the Poisson distribution. (Expression (16) of the paper)
        # The relation: λ ~ Ga(a, b), where a is the shape and b is rate (not scale)
        self.lambda_b = initial_lambda_b # Float64
        # Cache for easier computation
        # + 2 because of bow and eow
        self.whpylm_parent_p_w_cache = [0.0 for _ in range(3)] # OffsetVector{Float64}
        # Cache for the characters that make up the chant that was last added to the chpylm
        # Note that because this is a container that gets reused over and over again, its length is simply the maximum word length. The "actual word length" will be computed and passed in as a parameter to functions that use this variable. Not sure if this is the most sensible approach though. Maybe we can refactor it to a less tedious way later. The extra length parameter is no fun.
        self.most_recent_word = [] # OffsetVector{Char}

        self.sample_lambda_with_initial_params()

    def produce_word_with_bow_and_eow(chant_as_chars: list, word_begin_index: int, word_end_index: int):
        """
        Wraps a character array which represents a word with two special tokens: BOW and EOW
        I think a better idea is to just return an array each time which is never overly long.
        """
        # + 2 to accomodate BOW and EOW
        word = [' ' for _ in range(word_end_index - word_begin_index + 3)]
        word[0] = BOW
        # # The length is end - begin + 1. This is always the case.
        i = 0
        while i < (word_end_index - word_begin_index + 1):
            # - 1 because Julia arrays are 1-indexed
            word[i + 1] = chant_as_chars[word_begin_index + i]
            i += 1
        word[i + 1] = EOW
        return word

    def extend_capacity(self, max_chant_length: int):
        if (max_chant_length <= self.max_chant_length):
            return
        else:
            self.allocate_capacity(max_chant_length)

    def allocate_capacity(self, max_chant_length: int):
        self.max_chant_length = max_chant_length
        self.most_recent_word = [' ' for _ in range(max_chant_length + 2)]

    def sample_lambda_with_initial_params(self):
        for i in range(1, WORDTYPE_NUM_TYPES + 1):
            # scale = 1/rate
            self.lambda_for_types[i] = np.random.gamma(
                self.lambda_a,
                1/self.lambda_b
            )

    def add_customer_at_index_n(self, chant: "Chant", n: int) -> bool:
        """
        This function adds the nth segmented word in the chant to the NPYLM.
        """
        # The first two entries are always the BOS symbols.
        if not (n >= self.n_gram-1):
            raise Exception("n<n_gram-1")
        token_n: int = chant.get_nth_word_id(n)
        pyp: "PYP" = self.find_node_by_tracing_back_context_from_index_n_chant(chant, n, self.whpylm_parent_p_w_cache, True, False)
        if not pyp != None:
            raise Exception("pyp = None")
        num_tables_before_addition: int = self.whpylm.root.ntables
        index_of_table_added_to_in_root: list =[-1]
        pyp.add_customer(token_n, self.whpylm_parent_p_w_cache, self.whpylm.d_array, self.whpylm.theta_array, True, index_of_table_added_to_in_root)
        num_tables_after_addition: int = self.whpylm.root.ntables
        word_begin_index = chant.segment_begin_positions[n]
        word_end_index = word_begin_index + chant.segment_lengths[n] - 1
        # If the number of tables in the root is increased, we'll need to break down the word into characters and add them to the chpylm as well.
        # Remember that a customer has a certain probability to sit at a new table. However, it might also join an old table, in which case the G_0 doesn't change?
        if (num_tables_before_addition < num_tables_after_addition):
            # Because the CHPYLM is now modified, the cache is no longer valid.
            self.whpylm_G_0_cache = {}
            # EOS is not actually a "word". Therefore, it will always be set to be generated by the root node of the CHPYLM.
            if token_n == EOS:
                # Will need some sort of special treatment for EOS

                self.chpylm.root.add_customer(EOS_CHAR, self.chpylm.G_0, self.chpylm.d_array, self.chpylm.theta_array, True, index_of_table_added_to_in_root)
                return True
            if not (index_of_table_added_to_in_root[0] != -1):
                raise Exception(index_of_table_added_to_in_root[0] != -1)
            # Get the depths recorded for each table in the tablegroup of token_n.
            # It may not exist yet so we'll have to check and create it if that's the case.
            if not token_n in self.recorded_depth_arrays_for_tablegroups_of_token:
                self.recorded_depth_arrays_for_tablegroups_of_token[token_n] = []
            depth_arrays_for_the_tablegroup = self.recorded_depth_arrays_for_tablegroups_of_token[token_n]

            # This is a new table that didn't exist before *in the tablegroup for this token*.
            if not len(depth_arrays_for_the_tablegroup) <= index_of_table_added_to_in_root[0]:
                raise Exception("len(depth_arrays_for_the_tablegroup) > index_of_table_added_to_in_root[0]")
            # Variable to hold the depths of each character that was added to the CHPYLM as a part of the creation of this new table.
            # recorded_depth_array = Int[]
            # `word_end_index - word_begin_index + 2` is `length(word) - 1 + 2`, i.e. word_length_with_symbols
            recorded_depth_array = [0 for _ in range(word_end_index - word_begin_index + 3)]
            self.add_word_to_chpylm(chant.characters, word_begin_index, word_end_index, recorded_depth_array)
            if not (len(recorded_depth_array) == word_end_index - word_begin_index + 3):
                raise Exception("(len(recorded_depth_array)) != word_end_index-word_begin_index+3")
            # Therefore we push the result of depths for *this new table* into the array.
            depth_arrays_for_the_tablegroup.append(recorded_depth_array)
        return True
    def add_word_to_chpylm(self, chant_as_chars: list, word_begin_index: int, word_end_index: int, recorded_depths: list):
        """
        Yeah OK so token_ids is just a temporary variable holding all the characters to be added into the chpylm? What a weird design... 
        Why can't we do better let's see how we might refactor this code later.
        """
        if not word_end_index >= word_begin_index:
            raise Exception("word_end_index < word_begin_index")
        # This is probably to avoid EOS?
        if not word_end_index < self.max_chant_length:
            raise Exception("word_end_index >= self.max_chant_length")
        self.most_recent_word = NPYLM.produce_word_with_bow_and_eow(chant_as_chars, word_begin_index, word_end_index)
        # + 2 because of bow and eow
        word_length_with_symbols = word_end_index - word_begin_index + 1 + 2
        for n in range(word_length_with_symbols):
            depth_n = self.chpylm.sample_depth_at_index_n(self.most_recent_word, n, self.chpylm.parent_p_w_cache, self.chpylm.path_nodes)
            # println("depth_n sampled is $depth_n")
            self.chpylm.add_customer_at_index_n_npylm(self.most_recent_word, n, depth_n, self.chpylm.parent_p_w_cache, self.chpylm.path_nodes)
            # push!(recorded_depths, depth_n)
            recorded_depths[n] = depth_n

    def remove_customer_at_index_n(self, chant: "Chant", n: int):
        if not n >= 2:
            raise Exception("n<2")
        token_n = chant.get_nth_word_id(n)
        pyp:"PYP"= self.find_node_by_tracing_back_context_from_index_n_word_ids(chant.word_ids, n, False, False)
        if not pyp != None:
            raise Exception("pyp = None")
        num_tables_before_removal: int = self.whpylm.root.ntables
        index_of_table_removed_from = [-1]
        word_begin_index = chant.segment_begin_positions[n]
        word_end_index = word_begin_index + chant.segment_lengths[n] - 1

        # println("In remove_customer_at_index_n, before remove_customer. token_n: $token_n, word_begin_index: $word_begin_index, word_end_index: $word_end_index")
        pyp.remove_customer(token_n, True, index_of_table_removed_from)

        num_tables_after_removal: int = self.whpylm.root.ntables
        if num_tables_before_removal > num_tables_after_removal:
            # The CHPYLM is changed, so we need to clear the cache.
            self.whpylm_G_0_cache = {}
            if token_n == EOS:
                # EOS is not decomposable. It only gets added to the root node of the CHPYLM.
                # The char representation for EOS is what, "1"?
                self.chpylm.root.remove_customer(EOS_CHAR, True, index_of_table_removed_from)
                return True
            if not index_of_table_removed_from[0] != -1:
                raise Exception("index_of_table_removed_from[0] = -1")
            depths = self.recorded_depth_arrays_for_tablegroups_of_token[token_n]
            recorded_depths = depths[index_of_table_removed_from[0]]
            if not len(recorded_depths) > 0:
                raise Exception("len(recorded_depths) <= 0")
            self.remove_word_from_chpylm(chant.characters, word_begin_index, word_end_index, recorded_depths)
            # This entry is now removed.
            del depths[index_of_table_removed_from[0]]
        if pyp.need_to_remove_from_parent():
            pyp.remove_from_parent()
        return True

    def remove_word_from_chpylm(self, chant_as_chars: list, word_begin_index: int, word_end_index: int, recorded_depths: list):
        if not len(recorded_depths) > 0:
            raise Exception("len(recorded_depths) <= 0")
        if not word_end_index >= word_begin_index:
            raise Exception("word_end_index < word_begin_index")
        if not word_end_index < self.max_chant_length:
            raise Exception("word_end_index >= self.max_chant_length")
        self.most_recent_word = NPYLM.produce_word_with_bow_and_eow(chant_as_chars, word_begin_index, word_end_index)
        # + 2 because of bow and eow
        word_length_with_symbols = word_end_index - word_begin_index + 1 + 2
        if not len(recorded_depths) == word_length_with_symbols:
            raise Exception("len(recorded_depths) != word_length_with_symbols")
        for n in range(word_length_with_symbols):
            self.chpylm.remove_customer_at_index_n(self.most_recent_word, n, recorded_depths[n])

    def find_node_by_tracing_back_context_from_index_n_word_ids(self, word_ids: list, n: int, generate_if_not_found: bool, return_middle_node: bool) -> "PYP":
        # TODO: These all need to change when the bigram model is supported.
        if not n >= 2:
            raise Exception("n < 2")
        if not n < len(word_ids):
            raise Exception("n >= len(word_ids)")
        cur_node = self.whpylm.root
        # TODO: Why only 2?
        for depth in range(1, self.n_gram):
            # There are currently two BOS tokens.
            context = BOS
            if n - depth >= 0:
                context = word_ids[n - depth]
            child = cur_node.find_child_pyp(context, generate_if_not_found)
            if child == None:
                if return_middle_node:
                    return cur_node
                return None

            cur_node = child
        if not cur_node.depth == self.n_gram-1:
            raise Exception("cur_node.depth != n_gram-1")
        return cur_node

    def find_node_by_tracing_back_context_from_index_n_chant(self, chant: "Chant", n: int, parent_p_w_cache: list, generate_if_not_found: bool, return_middle_node: bool):
        """
        Used by add_customer
        """
        if not n >= self.n_gram-1:
            raise Exception("n < n_gram-1")
        # println("chant is $(chant), n is $(n), chant.num_segments is $(chant.num_segments)")
        if not n < chant.num_segments:
            raise Exception("n >= chant.num_segments")
        if not chant.segment_lengths[n] > 0:
            raise Exception("chant.segment_lengths[n] <= 0")
        word_begin_index = chant.segment_begin_positions[n]
        word_end_index = word_begin_index + chant.segment_lengths[n] - 1
        return self.find_node_by_tracing_back_context_from_index_n_both(chant.characters, chant.word_ids, n, word_begin_index, word_end_index, parent_p_w_cache, generate_if_not_found, return_middle_node) 

    def find_node_by_tracing_back_context_from_index_n_both(self, chant_as_chars: list, word_ids: list, n: int, word_begin_index: int, word_end_index: int, parent_p_w_cache: list, generate_if_not_found: bool, return_middle_node: bool)-> "PYP":
        """
        We should be filling the parent_p_w_cache while trying to find the node already. So if not there's some problem going on.
        """
        if not n >= self.n_gram-1:
            raise Exception("n < n_gram-1")
        if not n < len(word_ids):
            raise Exception("n >= len(word_ids)")
        if not word_begin_index >= 0:
            raise Exception("word_begin_index < 0")
        if not word_end_index >= word_begin_index:
            raise Exception("word_end_index < word_begin_index")
        cur_node = self.whpylm.root
        word_n_id = word_ids[n]
        parent_p_w = self.compute_G_0_of_word_at_index_n(chant_as_chars, word_begin_index, word_end_index, word_n_id)
        # println("The first parent_p_w is $parent_p_w")
        parent_p_w_cache[0] = parent_p_w
        for depth in range(1, self.n_gram):
            context = BOS
            if n - depth >= 0:
                context = word_ids[n - depth]
            # println("Trying to compute_p_w_with_parent_p_w, but what is word_n_id first??? $word_n_id")
            p_w = cur_node.compute_p_w_with_parent_p_w(word_n_id, parent_p_w, self.whpylm.d_array, self.whpylm.theta_array)
            # println("The depth is $depth, the p_w is $p_w")
            parent_p_w_cache[depth] = p_w
            child = cur_node.find_child_pyp(context, generate_if_not_found)
            if child == None and return_middle_node == True:
                return cur_node
            # So the other possibility will never be triggered?
            if not child != None:
                raise Exception("child = None")
            parent_p_w = p_w
            cur_node = child
        if not cur_node.depth == self.n_gram-1:
            raise Exception("cur_node.depth != n_gram-1")
        return cur_node

    def compute_G_0_of_word_at_index_n(self, chant_as_chars: list, word_begin_index: int, word_end_index: int, word_n_id) -> float:
        # println("In compute_G_0_of_word_at_index_n, chant_as_chars is $chant_as_chars, word_begin_index is $word_begin_index, word_end_index is $word_end_index, word_n_id is $word_n_id")
        if word_n_id == EOS:
            # println("The word is EOS, directly return")
            return self.chpylm.G_0

        if not word_end_index < self.max_chant_length:
            raise Exception("word_end_index >= self.max_chant_length")
        if not word_begin_index >= 0:
            raise Exception("word_begin_index < 0")
        if not word_end_index >= word_begin_index:
            raise Exception("word_end_index < word_begin_index")
        word_length = word_end_index - word_begin_index + 1
        # However, if the word does not exist in the cache, we'll then have to do the calculation anyways.
        if not word_n_id in self.whpylm_G_0_cache:
            # println("The nothing branch is entered.")
            # token_ids = npylm.most_recent_word
            self.most_recent_word = NPYLM.produce_word_with_bow_and_eow(chant_as_chars, word_begin_index, word_end_index)
            # Add bow and eow
            word_length_with_symbols = word_length + 2
            # p_w = compute_p_w(npylm.chpylm, token_ids, word_length_with_symbols)
            p_w = self.chpylm.compute_p_w(self.most_recent_word)
            # println("most_recent_word is $npylm.most_recent_word, p_w is $p_w")
            # If it's the very first iteration where there isn't any word yet, we cannot compute G_0 based on the chpylm.
            if word_length > self.max_word_length:
                self.whpylm_G_0_cache[word_n_id] = p_w
                return p_w
            else:
                # See section 4.3: p(k|Θ) is the probability that a word of *length* k will be generated from Θ, where Θ refers to the CHPYLM.
                p_k_given_chpylm = self.compute_p_k_given_chpylm(word_length)

                # Each word type will have a different poisson parameter
                t = detect_word_type_substr(chant_as_chars, word_begin_index, word_end_index)
                lmbda = self.lambda_for_types[t]
                # Deduce the word length with the Poisson distribution
                poisson_sample = NPYLM.sample_poisson_k_lambda(word_length, lmbda)
                if not poisson_sample > 0:
                    raise Exception("poisson_sample <= 0")
                # This is expression (15) of the paper, where we calculate p(c1...ck)
                # expression (5): p(c1...ck) returned from the CHPYLM is exactly the "G_0" of the WHPYLM, thus the naming.
                # G_0 is the more appropriate variable naming as this is just the way it's written in the expressions.
                G_0 = p_w / p_k_given_chpylm * poisson_sample

                # Very rarely the result will exceed 1
                if not (0 < G_0 and G_0 < 1):
                    # Now there is a bug and this branch is triggered all the time.
                    print("Very rarely the result will exceed 1")
                    for i in range(word_begin_index,word_end_index+1):
                        print(chant_as_chars[i])
                    print("\n")
                    print(p_w)
                    print(poisson_sample)
                    print(p_k_given_chpylm)
                    print(G_0)
                    print(word_length)
                self.whpylm_G_0_cache[word_n_id] = G_0
                return G_0
        else:
            # The cache already exists. No need for duplicated computation.
            # println("G_0 already exists in cache, it is $G_0")
            return self.whpylm_G_0_cache[word_n_id]

    def sample_poisson_k_lambda(k: int, lmbda: float) -> float:
        return poisson.pmf(k=k, mu=lmbda)

    def compute_p_k_given_chpylm(self, k: int) -> float:
        if k > self.max_word_length:
            return 0.0
        return self.p_k_chpylm[k]

    def sample_hyperparameters(self):
        self.whpylm.sample_hyperparameters()
        self.chpylm.sample_hyperparameters()

    def compute_probability_of_chant(self, chant: "Chant"):
        """
        Compute the probability of the chant by using the product of the probabilities of the words that make up the chant.
        """
        prod = 1.0
        for n in range(2, chant.num_segments):
            prod *= self.compute_p_w_of_nth_word(self, chant, n)
        return prod

    def compute_log_probability_of_chant(self, chant: "Chant"):
        """
        Compute the probability of the chant by using the sum of the log probabilities of the words that make up the chant.
        Using log could be more beneficial in preventing underflow.
        """
        sum = 0.0
        for n in range(2, chant.num_segments):
            sum += np.log(self.compute_p_w_of_nth_word(chant, n))
        return sum

    def compute_p_w_of_nth_word(self, chant: "Chant", n: int):
        """
        This is the real "compute_p_w"... The above ones don't have much to do with p_w I reckon. They are about whole chants. Eh.
        """
        if not n >= 2:
            raise Exception("n < 2")
        if not n < chant.num_segments:
            raise Exception("n >= chant.num_segments")
        if not chant.segment_lengths[n] > 0:
            raise Exception("chant.segment_lengths[n] <= 0")
        word_begin_index = chant.segment_begin_positions[n]
        # I mean, why don't you just record the end index directly anyways. The current implementation is such a torture.
        word_end_index = word_begin_index + chant.segment_lengths[n] - 1
        return self.compute_p_w_of_nth_word_chars(chant.characters, chant.word_ids, n, word_begin_index, word_end_index)


    def compute_p_w_of_nth_word_chars(self, chant_as_chars: list, word_ids: list, n: int, word_begin_position: int, word_end_position: int) -> float:
        word_id = word_ids[n]
        # println("We're in compute_p_w_of_nth_word, chant_as_chars: $chant_as_chars, word_ids: $word_ids, word_begin_index: $word_begin_position, word_end_index: $word_end_position")

        # So apparently the parent_p_w_cache should be set while we're trying to find the node?
        # generate_if_not_found = false, return_middle_node = true
        node = self.find_node_by_tracing_back_context_from_index_n_both(chant_as_chars, word_ids, n, word_begin_position, word_end_position, self.whpylm_parent_p_w_cache, False, True)
        if not node != None:
            raise Exception("node = None")
        # println("Node is $node")
        parent_p_w = self.whpylm_parent_p_w_cache[node.depth]
        # println("The parent_p_w is $parent_p_w")
        # The final `true` indicates that it's the with_parent_p_w variant of the function
        # println("What happened?")
        return node.compute_p_w_with_parent_p_w(word_id, parent_p_w, self.whpylm.d_array, self.whpylm.theta_array)