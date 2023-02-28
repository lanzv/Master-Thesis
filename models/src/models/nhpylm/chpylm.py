from src.models.nhpylm.hpylm import HPYLM
from src.models.nhpylm.pyp import PYP
from src.models.nhpylm.definitions import BOW, CHPYLM_EPSILON, EOW, HPYLM_INITIAL_D, HPYLM_INITIAL_THETA, HPYLM_A, HPYLM_B, HPYLM_ALPHA, HPYLM_BETA
import numpy as np


class CHPYLM(HPYLM):
    def __init__(self, g_0: float, max_depth: int, beta_stop: float, beta_pass: float):
        super().__init__()
        # All the fields are "inherited" from HPYLM.
        self.root = PYP(BOW)
        self.depth: int = 0
        self.G_0: float = g_0
        self.d_array:list = []
        self.theta_array:list = []

        # These variables are related to the sampling process as described in the Teh technical report, expressions (40) and (41)
        # Note that they do *not* directly correspond to the alpha, beta parameters of a Beta distribution, nor the shape and scale parameters of a Gamma distribution.
        # For the sampling of discount d
        self.a_array = []
        # For the sampling of discount d
        self.b_array = []
        # For the sampling of concentration θ
        self.alpha_array = []
        # For the sampling of concentration θ
        self.beta_array =[]

        # Fields specific to CHPYLM
        self.beta_stop: float = beta_stop
        self.beta_pass: float = beta_pass
        self.max_depth: int = max_depth
        self.parent_p_w_cache: list = [0.0 for _ in range(max_depth)]
        self.path_nodes: list = [None for _ in range(max_depth)] #Vec<Option<*mut PYP<char>>>,

    def add_customer_at_index_n(self, string_as_chars: list, n: int, depth: int) -> bool:
        """
        The sampling process for the infinite Markov model is similar to that of the normal HPYLM in that you
        - first remove the nth customer which resides at the depth "order-of-nth-customer", *decrementing* pass_count or stop_count along the path of the tree
        - sample a new order (depth) according to the conditional probability
        - add this (originally nth) customer back again at the newly sampled depth, *incrementing* pass_count or stop_count along the (new) path
        This function adds the customer
        """
        node = self.find_node_by_tracing_back_context(string_as_chars, n, depth, self.parent_p_w_cache)
        char_n = string_as_chars[n]
        root_table_index = [0]
        return node.add_customer(node, char_n, self.parent_p_w_cache, self.d_array, self.theta_array, True, root_table_index)
    
    def add_customer_at_index_n_npylm(self, characters: list, n: int, depth: int, parent_p_w_cache: list, path_nodes: list) -> bool:
        """
        This function adds the customer. See documentation above.
        This is a version to be called from the NPYLM.
        If the parent_p_w_cache is already set, then update the path_nodes as well.
        """
        assert(0 <= depth and depth <= n)
        node = self.find_node_by_tracing_back_context_npylm(characters, n, depth, path_nodes)
        # Seems to be just a check
        if depth > 0:
            if (node.context != characters[n - depth]):
                print("node.context is $(node.context), characters[n - depth] is $(characters[n - depth]), characters is $characters, n is $n, depth is $depth")
            # @assert(node.context == characters[n - depth])
        assert(node.depth == depth)
        char_n = characters[n]
        root_table_index = [0]
        return node.add_customer(char_n, parent_p_w_cache, self.d_array, self.theta_array, True, root_table_index)

    def remove_customer_at_index_n(self, characters: list, n: int, depth: int) -> bool:
        """
        The sampling process for the infinite Markov model is similar to that of the normal HPYLM in that you
        - first remove the nth customer which resides at the depth "order-of-nth-customer", *decrementing* pass_count or stop_count along the path of the tree
        - sample a new order (depth) according to the conditional probability
        - add this (originally nth) customer back again at the newly sampled depth, *incrementing* pass_count or stop_count along the (new) path
        This function removes the customer
        """
        assert(0 <= depth and depth <= n)
        node = self.find_node_by_tracing_back_context_remove(characters, n, depth, False, False)
        assert node != None
        # Seems to be just a check
        if depth > 0:
            assert(node.context == characters[n - depth])
        assert(node.depth == depth)
        char_n = characters[n]
        root_table_index = [0]
        node.remove_customer(char_n, True, root_table_index)

        # Check if the node needs to be removed
        if node.need_to_remove_from_parent():
            node.remove_from_parent()
        
        return True

    def find_node_by_tracing_back_context_remove(self, characters: list, n: int, depth_of_n: int, generate_if_not_found: bool, return_cur_node_if_not_found: bool) -> "PYP":
        """
        For the nth customer, this function finds the node with depth `depth_of_n` in the suffix tree.
        The found node contains the correct contexts of length `depth_of_n` for the nth customer.
        Example:
        [h,e,r, ,n,a,m,e]
        n = 3
        depth_of_n = 2
        The customer is "r". With a `depth_of_n` of 2, We should get the node for "h".
        When we connect the node all the way up, we can reconstruct the full 2-gram context "h-e-" that is supposed to have generated the customer "r".
        This version is used during `remove_customer`.
        """
        # This situation makes no sense, otherwise we'll go straight out of the start of the word.
        if n < depth_of_n:
            return None
        # Note that we start from the root.
        cur_node = self.root
        for d in range(1, depth_of_n+1):
            context = characters[n - d]
            # Find the child pyp whose context is the given context
            child:"PYP" = cur_node.find_child_pyp(context, generate_if_not_found)
            if child == None:
                if return_cur_node_if_not_found:
                    return cur_node
                else:
                    return None
            else:
                # Then, using that child pyp as the starting point, find its child which contains the context one further back again.
                cur_node = child

        # The search has ended for the whole depth.
        # In this situation the cur_node should have the same depth as the given depth.
        assert(cur_node.depth == depth_of_n)
        if depth_of_n > 0:
            assert(cur_node.context == characters[n - depth_of_n])
        return cur_node

    def find_node_by_tracing_back_context(self, characters: list, n: int, depth_of_n: int, parent_p_w_cache: list) -> "PYP":
        """
        For the nth customer, this function finds the node with depth `depth_of_n` in the suffix tree.
        The found node contains the correct contexts of length `depth_of_n` for the nth customer.
        Example:
        [h,e,r, ,n,a,m,e]
        n = 3
        depth_of_n = 2
        The customer is "r". With a `depth_of_n` of 2, We should get the node for "h".
        When we connect the node all the way up, we can reconstruct the full 2-gram context "h-e-" that is supposed to have generated the customer "r".
        This version is used during `add_customer`. It cachees the probabilities of generating the nth customer at each level of the tree, during the tracing.
        """
        # This situation makes no sense, otherwise we'll go straight out of the start of the word.
        if n < depth_of_n:
            return None

        # The actual char at location n of the chant
        char_n = characters[n]
        # Start from the root node, order 0
        cur_node = self.root
        parent_p_w = self.G_0
        parent_p_w_cache[0] = parent_p_w
        for depth in range(1, depth_of_n+1):
            # What is the possibility of char_n being generated from cur_node (i.e. having cur_node as its context).
            p_w = cur_node.compute_p_w_with_parent_p_w(char_n, parent_p_w, self.d_array, self.theta_array)
            parent_p_w_cache[depth] = p_w

            # The context `depth`-order before the target char
            context_char = characters[n - depth]
            # We should be able to find the PYP containing that context char as a child of the current node. If it doesn't exist yet, create it.
            child = cur_node.find_child_pyp(context_char, True)
            parent_p_w = p_w
            cur_node = child
        return cur_node

    def find_node_by_tracing_back_context_npylm(self, characters: list, n: int, depth_of_n: int, path_nodes_cache: list) -> "PYP":
        # This situation makes no sense, otherwise we'll go straight out of the start of the word.
        if n < depth_of_n:
            return None
        cur_node = self.root
        for depth in range(depth_of_n):
            # + 1 because we're always looking at the path node, i.e. the node one level higher up.
            # path_node = get(path_nodes_cache, depth + 1, nothing)
            # if path_node != nothing
            if path_nodes_cache[depth + 1] != None:
                cur_node = path_nodes_cache[depth + 1]
            else:
                context_char = characters[n - depth - 1]
                child = cur_node.find_child_pyp(context_char, True)
                cur_node = child
        return cur_node

    def compute_p_w(self, characters: list) -> float:
        """
        Compute the probability of a word (represented as an OffsetVector{Char}) in this CHPYLM.
        """
        return np.exp(self.compute_log_p_w(characters))
    
    def compute_log_p_w(self, characters: list):
        """
        Compute the *log* probability of a word (represented as an OffsetVector{Char}) in this CHPYLM.
        """
        char = characters[0]
        log_p_w = 0.0
        # I still haven't fully wrapped my head around the inclusions and exclusions of BOS, EOS, BOW, EOW, etc. Let's see how this works out though.
        if char != BOW:
            log_p_w += np.log(self.root.compute_p_w(char, self.G_0, self.d_array, self.theta_array))

        for n in range(1, len(characters)):
            # I sense that the way this calculation is written is simply not very efficient. Surely we can do better than this?
            # n - 1 because that argument is the end of the context `h`, not the actual word itself.
            # I sense another indexing error previously here. Why would it start from 0 instead of 1?
            log_p_w += np.log(self.compute_p_w_given_h(characters, 0, n - 1))

        return log_p_w
    
    def compute_p_w_given_h(self, characters, context_begin: int, context_end: int) -> float:
        """
        Compute the probability of generating the character `characters[end + 1]` with `characters[begin:end]` as the context, with this CHPYLM.
        """
        target_char = characters[context_end] #+1
        return self.compute_p_w_given_target_char_and_h(target_char, characters, context_begin, context_end)

    def compute_p_w_given_target_char_and_h(self, target_char, characters: list, context_begin: int, context_end: int):
        cur_node = self.root
        parent_pass_probability = 1.0
        p = 0.0
        # We start from the root of the tree.
        parent_p_w = self.G_0
        p_stop = 1.0
        depth = 0

        # There might be calculations that result in depths greater than the actual context length.
        while (p_stop > CHPYLM_EPSILON):
            # If there is no node yet, use the Beta prior to calculate
            if cur_node == None:
                p_stop = (self.beta_stop) / (self.beta_pass + self.beta_stop) * parent_pass_probability
                p += parent_p_w * p_stop
                parent_pass_probability *= (self.beta_pass) / (self.beta_pass + self.beta_stop)
            else:
                p_w = cur_node.compute_p_w_with_parent_p_w(target_char, parent_p_w, self.d_array, self.theta_array)
                p_stop = cur_node.stop_probability(self.beta_stop, self.beta_pass, False) * parent_pass_probability
                p += p_w * p_stop
                parent_pass_probability *= cur_node.pass_probability(self.beta_stop, self.beta_pass, False)
                parent_p_w = p_w

                # Preparation for the next round.
                # We do this only in the else branch because if the cur_node is already `nothing`, it will just keep being `nothing` from then onwards.

                # If wee've already gone so deep that the depth is greater than the actual context length. Of course there's no next node at such a depth.
                # Note that this operation is with regards to the next node, thus the + 1 on the left hand side.
                # On the right hand side the + 1 is because we're getting the length, which requires + 1
                # So the two +1's are not the same!
                if depth + 1 >= context_end - context_begin + 1:
                    cur_node = None
                else:
                    cur_context_char = characters[context_end - depth]
                    child = cur_node.find_child_pyp(cur_context_char)
                    cur_node = child
            depth += 1
        assert p > 0.0
        return p

    def sample_depth_at_index_n(self, characters: list, n: int, parent_p_w_cache: list, path_nodes: list) -> int:
        """
        Sample the depth of the character at index n of the given characters (word).
        """
        # The first character should always be the BOW
        if (n == 0):
            return 0
        
        # Make sure that sampling_table has the right length
        sampling_table = [0.0 for _ in range(n+1)] # OffsetVector{Float64}(undef, 0:n)
        char_n = characters[n]
        sum = 0.0
        parent_p_w = self.G_0
        parent_pass_probability = 1.0
        parent_p_w_cache[0] = parent_p_w
        sampling_table_size = 0
        cur_node = self.root
        for index in range(n+1):
            # Already gone beyond the original word's context length.
            if cur_node == None:
                p_stop = (self.beta_stop) / (self.beta_pass + self.beta_stop) * parent_pass_probability
                # If there's already no new context char, we just use the parent word probability as it is.
                p = parent_p_w * p_stop
                # The node with depth n is the parent of the node with depth n + 1. Therefore the index + 1 here.
                parent_p_w_cache[index + 1] = parent_p_w
                sampling_table[index] = p
                path_nodes[index] = None
                sampling_table_size += 1
                sum += p
                parent_pass_probability *= (self.beta_pass) / (self.beta_pass + self.beta_stop)
                if (p_stop < CHPYLM_EPSILON):
                    break
            else:
                p_w = cur_node.compute_p_w_with_parent_p_w(char_n, parent_p_w, self.d_array, self.theta_array)
                p_stop = cur_node.stop_probability(self.beta_stop, self.beta_pass, False)
                p = p_w * p_stop * parent_pass_probability
                parent_p_w = p_w
                parent_p_w_cache[index + 1] = parent_p_w
                sampling_table[index] = p
                path_nodes[index] = cur_node
                sampling_table_size += 1
                parent_pass_probability *= cur_node.pass_probability(self.beta_stop, self.beta_pass, False)
                sum += p
                if (p_stop < CHPYLM_EPSILON):
                    break
                if index < n:
                    context_char = characters[n - index - 1]
                    cur_node = cur_node.find_child_pyp(context_char)

        # The following samples the depth according to their respective probabilities.
        depths = np.arange(0, len(sampling_table))
        ps = np.array(sampling_table)/np.array(sampling_table).sum()
        return np.random.choice(depths, p=ps)

    # Implementation of HPYLM functions

    def get_num_nodes(self) -> int:
        return self.root.get_num_nodes() + 1

    def get_num_tables(self) -> int:
        return self.root.get_num_tables() + 1

    def get_num_customers(self) -> int:
        return self.root.get_num_customers() + 1

    def get_pass_counts(self) -> int:
        return self.root.get_pass_counts() + 1

    def get_stop_counts(self) -> int:
        return self.root.get_stop_counts() + 1

    def sample_hyperparameters(self):
        max_depth = len(self.d_array) - 1
        sum_log_x_u_array = [0.0 for _ in range(max_depth + 1)]
        sum_y_ui_array = [0.0 for _ in range(max_depth + 1)]
        sum_one_minus_y_ui_array = [0.0 for _ in range(max_depth + 1)]
        sum_one_minus_z_uwkj_array = [0.0 for _ in range(max_depth + 1)]

        self.depth = 0
        self.depth = self.sum_auxiliary_variables_recursively(self.root, sum_log_x_u_array, sum_y_ui_array, sum_one_minus_y_ui_array, sum_one_minus_z_uwkj_array, self.depth)
        self.init_hyperparameters_at_depth_if_needed(self.depth)

        for u in range(self.depth):
            self.d_array[u] = np.random.beta(
                self.a_array[u] + sum_one_minus_y_ui_array[u],
                self.b_array[u] + sum_one_minus_z_uwkj_array[u])
            self.theta_array[u] = np.random.gamma(
                self.alpha_array[u] + sum_y_ui_array[u],
                1.0 / (self.beta_array[u] - sum_log_x_u_array[u])
            )

        excessive_length = max_depth - self.depth
        for _ in range(excessive_length):
            self.d_array.pop()
            self.theta_array.pop()
            self.a_array.pop()
            self.b_array.pop()
            self.alpha_array.pop()
            self.beta_array.pop()