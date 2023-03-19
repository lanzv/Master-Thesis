from src.models.nhpylm.corpus import Dataset, Vocabulary
from src.models.nhpylm.chant import Chant
from src.models.nhpylm.wtype import WORDTYPE_NUM_TYPES, detect_word_type
from src.models.nhpylm.definitions import EOW, BOW
import numpy as np
"""
Actually I'm not sure if we really need such a complicated Trainer class. Let's first go on though.
This struct contains everything needed for the training process
"""
class Trainer:
    def __init__(self, dataset: "Dataset", model, always_accept_new_segmentation: bool = True):
        self.rand_indices_train: list = [i for i in range(len(dataset.train_chants))]
        self.rand_indices_dev: list =  [i for i in range(len(dataset.dev_chants))]
        self.dataset: "Dataset" = dataset
        self.vocabulary: "Vocabulary" = dataset.vocabulary
        self.model = model
        # These tables are used when we generate words randomly from the CHPYLM, in the `sample_next_char_from_chpylm_given_context` function.
        self.chpylm_sampling_probability_table: list = [0.0 for _ in range(dataset.vocabulary.get_num_characters() + 1)] # Vector{Float64}
        self.chpylm_sampling_id_table: list = [' ' for _ in range(dataset.vocabulary.get_num_characters() + 1)] # Vector{Char}
        self.always_accept_new_segmentation: bool = always_accept_new_segmentation
        # Indicates whether the chant at this index has already been added to the CHPYLM. If yes, in iterations > 2 we'd need to remove the chant from the CHPYLM and add it again.
        self.added_to_chpylm_train: list = [False for _ in range(len(dataset.train_chants))] # Vector{Bool}
        # If we don't always accept new segmentations, some segmentations might be rejected.
        self.num_segmentation_rejections: int = 0
        self.num_segmentation_acceptances: int = 0

    def sample_hyperparameters(self):
        self.model.npylm.sample_hyperparameters()

    def sample_lambda(self):
        """
        Sample lambda values for different types of characters.
 
        For example, puncutation marks, alphabets, Chinese ideographs are all different types of characters.
        Each type would get its own average word length correction with a different lambda value.
        """
        a_array = [self.model.sampler.npylm.lambda_a for _ in range(WORDTYPE_NUM_TYPES + 1)]
        b_array = [self.model.sampler.npylm.lambda_b for _ in range(WORDTYPE_NUM_TYPES + 1)]
        word_ids: set = set()
        # Get all chants in the training set.
        for chant in self.dataset.train_chants:
            # Go through each word in the chant, excluding the BOS and EOS tokens.
            for index in range(2, chant.num_segments - 1):
                word = chant.get_nth_word_string(index)
                word_id = chant.get_nth_word_id(index)
                word_length = chant.get_nth_segment_length(index)
                if word_length > self.model.npylm.max_word_length:
                    continue

                # If the word hasn't been added to the set of known words yet, add it.
                if not word_id in word_ids:
                    # Get the tablegroups that correspond to this word in the root of the WHPYLM. Essentially we're just trying to count how frequent this word appeared.
                    # IMO this word should always be present in the root of the WHPYLM. If not then it's a bug. Anyways the [] there is just a failsafe measure.
                    if word_id in self.model.npylm.whpylm.root.tablegroups:
                        tablegroups = self.model.npylm.whpylm.root.tablegroups[word_id]
                    else:
                        tablegroups = []
                    num_tablegroups = len(tablegroups)
                    t = detect_word_type(word)
                    a_array[t] += num_tablegroups * word_length
                    b_array[t] += num_tablegroups
                    word_ids.add(word_id)

            for t in range(1, WORDTYPE_NUM_TYPES+1):
                self.model.npylm.lambda_for_types[t] = np.random.gamma(
                    a_array[t],
                    1 / b_array[t]
                )

    def sample_next_char_from_chpylm_given_context(self, context_chars: list, context_length: int, sample_t: int, skip_eow: bool):
        """
        This function tries to generate a word randomly from the CHPYLM. Used by the function `update_p_k_given_chpylm`.
        `skip_eow` means that EOW shouldn't be generated as the next char. This applies when there is only BOW in the current word so far.
        """
        prob_sum = 0.0
        chpylm = self.model.npylm.chpylm
        table_index = 0
        all_characters = self.vocabulary.all_characters
        num_characters = len(all_characters)
        for c in all_characters:
            # context_begin: 0, context_end: length - 1
            p_w = chpylm.compute_p_w_given_target_char_and_h(c, context_chars, 0, context_length - 1)
            prob_sum += p_w
            self.chpylm_sampling_probability_table[table_index] = p_w
            self.chpylm_sampling_id_table[table_index] = c
            table_index += 1

        # Also record EOW as a probable character to be sampled.
        if not skip_eow:
            p_w = chpylm.compute_p_w_given_target_char_and_h(EOW, context_chars, 0, context_length - 1)
            prob_sum += p_w
            self.chpylm_sampling_probability_table[table_index] = p_w
            self.chpylm_sampling_id_table[table_index] = EOW

        # Sample one character from the table.
        chosen_index = np.random.choice(np.arange(0, len(self.chpylm_sampling_id_table)),
                p=self.chpylm_sampling_probability_table/np.array(self.chpylm_sampling_probability_table).sum())
        return self.chpylm_sampling_id_table[chosen_index]



    def update_p_k_given_chpylm(self, num_samples: int = 20000, early_stopping_threshold: int = 10):
        """
        This function updates the cache of the probability of sampling a word of length k from the CHPYLM.
        As mentioned in Section 4.3 of the paper, a Monte Carlo method is employed to generate words randomly from the CHPYLM so that empirical estimates of p(k|chpylm) can be obtained.
        """
        max_word_length = self.model.get_max_word_length() + 1
        p_k_chpylm = self.model.npylm.p_k_chpylm
        # Do you mean num_characters. Eh.
        # It's 1 longer than the original max_word_length, probably we have 0 in order to incorporate the possibility of getting length 0 word?
        # This array keeps track of total numbers of words of length k.
        # max_word_length + 1 because also a special case of k > max_word_length needs to be tracked?
        # Note that we need to provide a type argument to zeros in this case.
        num_words_of_length_k = [0 for _ in range(max_word_length + 1)]
        for i in range(max_word_length+1):
            p_k_chpylm[i] = 0.0

        wrapped_chars = [' ' for _ in range(max_word_length + 3)]
        num_words_sampled = 0
        for itr in range(1, num_samples+1):
            wrapped_chars[0] = BOW

            # Keeps track of the actual word length
            cur_word_length = 0
            for j in range(max_word_length):
                # If we're only at the beginning of the chant we shouldn't sample an EOW, because that would result in the "word" containing no characters at all.
                skip_eow = True if (j == 0) else False
                next_char = self.sample_next_char_from_chpylm_given_context(wrapped_chars, j + 1, j + 1, skip_eow)
                wrapped_chars[j + 1] = next_char
                # EOW means the word is completely sampled.
                if next_char == EOW:
                    break
                cur_word_length += 1

            num_words_sampled += 1

            # In this case we just sampled an empty word, i.e. <BOW><EOW>. It cannot be used. Continue to the next round of sampling.
            if cur_word_length == 0:
                continue

            if not cur_word_length <= max_word_length:
                raise Exception("cur_word_length > max_word_length")
            num_words_of_length_k[cur_word_length] += 1

            # If all possible lengths have enough data generated, we can terminate the sampling early.
            if itr % 100 == 0:
                can_stop = True
                for k in range(1, max_word_length + 1):
                    if num_words_of_length_k[k] < early_stopping_threshold:
                        can_stop = False
                        break
                if can_stop:
                    break

        for k in range(1, max_word_length + 1):
            # Put in a Laplace smoothing over the final figures. Though seems that the divisor doesn't need this treatment anyways.
            # p_k_chpylm[k] = (num_words_of_length_k[k] + 1) / (num_words_sampled + max_word_length + 1)
            p_k_chpylm[k] = (num_words_of_length_k[k] + 1) / (num_words_sampled + max_word_length)
            if not p_k_chpylm[k] > 0:
                raise Exception("p_k_chpylm[k] <= 0")


    def blocked_gibbs_sampling(self):
        # Yeah I think we're not doing any segmentation in the first round at all. Segmentations only start from the second round. So the behavior is normal.
        # Still then the problem is why on only certain chants they try to remove EOS twice from the table. Fucking hell this just simply doesn't make the least bit of sense whatsoever let's just go on and see then.
        # temp_chant = trainer.dataset.train_chants[1]
        # println("In blocked_gibbs_sampling, temp_chant is $temp_chant, temp_chant.num_segments is $(temp_chant.num_segments), temp_chant.segment_lengths is $(temp_chant.segment_lengths) ")
        num_chants = len(self.dataset.train_chants)
        max_chant_length = self.dataset.max_chant_length

        # TODO: ... Why don't you just shuffle the array of chants itself instead of this seemingly extraneous array of indices?
        np.random.shuffle(self.rand_indices_train)

        # Update model parameters
        for step in range(num_chants):
            chant_index = self.rand_indices_train[step]
            chant: "Chant" = self.dataset.train_chants[chant_index]

            if chant.supervised:
                # Remove the segmentation and add it again, so that the seating arrangements can be updated.
                if self.added_to_chpylm_train[chant_index] == True:
                    for n in range(2, chant.num_segments):
                        self.model.npylm.remove_customer_at_index_n(chant, n)
                # Because this is supervised data, i.e. guaranteed to be the true segmentation, we don't need to resample the chant at all.
                for n in range(2, chant.num_segments):
                    self.model.npylm.add_customer_at_index_n(chant, n)
                self.added_to_chpylm_train[chant_index] = True
                continue
            else:
                # TODO: I thought this has more to do with the iteration of sampling? Do we really need such a mechanism anyways. But where is the iteration number in the first place eh.
                if self.added_to_chpylm_train[chant_index] == True:
                    old_segment_lengths = [0 for _ in range(max_chant_length + 3)]
                    num_old_segments = 0
                    old_log_p_s = 0.0
                    new_log_p_s = 0.0

                    # Wait, why is this thing triggered in the first round already. Even this doesn't seem to make sense.
                    for n in range(2, chant.num_segments):
                        # println("In blocked_gibbs_sampling, n is $n, chant is $chant, chant.num_segments is $(chant.num_segments), chant.segment_lengths is $(chant.segment_lengths) ")
                        self.model.npylm.remove_customer_at_index_n(chant, n)

                    # We need to later decide by some criteria whether to accept the new segmentation or just keep the old one.
                    if self.always_accept_new_segmentation == False:
                        num_old_segments = chant.get_num_segments_without_special_tokens()
                        for i in range(num_old_segments):
                            # We save the old segmentation but get rid of the BOS and EOS tokens
                            # Two BOS in the beginning.
                            old_segment_lengths[i] = chant.segment_lengths[i + 2]
                        old_log_p_s = self.model.npylm.compute_log_probability_of_chant(chant)

                    # Produce the new segmentation
                    new_segment_lengths = self.model.sampler.blocked_gibbs_segment(chant, True)
                    # println("new_segment_lengths is $new_segment_lengths")
                    chant.split_chant(new_segment_lengths)

                    # TODO: There might be a way to avoid performing the check twice? Using a single Chant struct to hold all these stuffs is quite a bit restrictive.
                    if self.always_accept_new_segmentation == False:
                        new_log_p_s = self.model.npylm.compute_log_probability_of_chant(chant)
                        # When the log probability of the new segmentation is lower, accept the new segmentation only with a certain probability
                        bernoulli = min(1.0, np.exp(new_log_p_s - old_log_p_s))
                        r = np.random.uniform(0,1)
                        if bernoulli < r:
                            chant.split_chant(old_segment_lengths, num_old_segments)
                            self.num_segmentation_rejections += 1
                        else:
                            self.num_segmentation_acceptances += 1


                # Put the chant data into the NPYLM
                # Yeah I think I get it. All the sampling process we're basically trying to alter the model parameters. We're not really storing the segmentation results of the training chants anyways, as that would be quite pointless and irrelevant. Let's go then.
                for n in range(2, chant.num_segments):
                    self.model.npylm.add_customer_at_index_n(chant, n)
                self.added_to_chpylm_train[chant_index] = True
        if not self.model.npylm.whpylm.root.ntables <= self.model.npylm.chpylm.get_num_customers():
            raise Exception("self.model.npylm.whpylm.root.ntables > self.model.npylm.chpylm.get_num_customers()")


    def compute_perplexity(self, chants: list):
        """
        TODO: Summarize the difference between the usgae of the Viterbi algorithm and the original blocked sampler.
        Compute the perplexity based on optimal segmentation produced by the Viterbi algorithm"
        """
        num_chants = len(chants)
        if num_chants == 0:
            return 0.0
        sum = 0.0

        for s in chants:
            # Create a copy so that no interference occurs.
            chant: "Chant" = Chant(s.chant_string)
            segment_lengths = self.model.sampler.viterbi_decode(chant)
            chant.split_chant(segment_lengths)
            # Why - 2 not - 3 though? EOS still needs to be taken into account in perplexity computation I guess?
            sum += self.model.npylm.compute_log_probability_of_chant(chant) / chant.num_segments - 2
        ppl = np.exp(-sum / num_chants)
        return ppl

    def compute_perplexity_train(self):
        return self.compute_perplexity(self.dataset.train_chants)

    def compute_perplexity_dev(self):
        return self.compute_perplexity(self.dataset.dev_chants)

    def compute_log_likelihood(self, chants: list):
        num_chants = len(chants)
        if num_chants == 0:
            return 0.0
        sum = 0.0
        for chant in chants:
            log_p_x = self.model.sampler.compute_log_forward_probability(chant, True)
            sum += log_p_x
        return sum

    def compute_log_likelihood_train(self):
        return self.compute_log_likelihood(self.dataset.train_chants)

    def compute_log_likelihood_dev(self):
        return self.compute_log_likelihood(self.dataset.dev_chants)

    def print_segmentations(self, num_to_print: int, chants: list, rand_indices: list):
        num_to_print = min(len(chants), num_to_print)
        for n in range(1, num_to_print + 1):
            chant_index = rand_indices[n]
            # I think this should really be clone, not just create a new chant based on the string. Let's see then.
            # OK I don't think it makes a difference because the chant provided to it as trainer.dataset.train_chants and trainer.dataset.dev_chants are probably still non-segmented anyways.
            chant = Chant(chants[chant_index].chant_string)
            # Use the viterbi_decode method to segment chants, given an already trained model.
            segment_lengths = self.model.sampler.viterbi_decode(chant)
            chant.split_chant(segment_lengths)
            chant.show()

    def print_segmentations_train(self, num_to_print: int):
        return self.print_segmentations(num_to_print, self.dataset.train_chants, self.rand_indices_train)

    def print_segmentations_dev(self, num_to_print: int):
        # shuffle!(trainer.rand_indices_dev) Rust doesn't have this part of code, only Julia does
        return self.print_segmentations(num_to_print, self.dataset.dev_chants, self.rand_indices_dev)