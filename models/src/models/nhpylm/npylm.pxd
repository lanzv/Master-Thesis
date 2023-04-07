from chant cimport Chant
import numpy as np
cimport numpy as np

cdef class NPYLM():
    cdef int max_segment_size
    cdef float beta_stops, beta_passes
    cdef int shpylm_max_depth, thpylm_max_depth
    cdef list shpylm_ds, shpylm_thetas, thpylm_ds, thpylm_thetas
    cdef float init_d, init_theta, d_a, d_b, theta_alpha, theta_beta
    cdef float theta
    cdef float poisson_lambda
    cdef float init_poisson_a
    cdef float init_poisson_b
    cdef list poisson_k_probs
    cdef set tone_vocabulary
    cdef dict root_tables_context_lengths
    cdef int last_removed_shpylm_table_index
    cdef THPYLMNode thpylm_root # root of tone HPYLM tree
    cdef SHPYLMNode shpylm_root # root of segment HPYLM tree
    cdef void add_chant(self, Chant chant)
    cdef void remove_chant(self, Chant chant)
    cdef float get_bigram_probability(self, str first_gram, str second_gram)
    cdef float get_segmentation_log_probability(self, list chant_segmentation)
    cdef void add_segment_to_thpylm(self, str segment)
    cdef void remove_segment_from_thpylm(self, str segment)
    cdef float get_G0_probability(self, str segment)



# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX          Segment Hierarchical Pitman-Yor Tree Node          XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

cdef class SHPYLMNode():
    cdef int depth
    cdef SHPYLMNode parent
    cdef NPYLM npylm
    cdef str hpylm_type # 't' for tone hpylm, 's' for shant hpylm
    cdef dict children # dictionary of deeper context HPYLNodes
    cdef str context # one previous element
    cdef dict tables # dict of all tables - key: segment, value: Tables
    cdef dict c_wh # c(w|h) .. count of all customers 
    cdef int c_h # sum of all counts in c(w|h)
    cdef int t_h # t_{h} .. sum of counts of tables in tables dictionary
    cdef bint add_segment(self, str segment, list context, float pwhcomma)
    cdef bint remove_segment(self, str segment, list context)
    cdef float get_pwh_probability(self, str segment, list context, float pwhcomma)


cdef class STables():
    # t_hw could be get as len(self.tables)
    cdef list tables # list of ints - customer counts
    cdef str segment
    cdef SHPYLMNode hpylmnode
    cdef bint add_customer(self, float pwhcomma, int t_h)
    cdef bint remove_customer(self)


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX           Tone Hierarchical Pitman-Yor Tree Node           XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

cdef class THPYLMNode():
    cdef THPYLMNode parent
    cdef NPYLM npylm
    cdef int depth
    cdef int stops # stop counts
    cdef int passes # pass counts
    cdef str hpylm_type # 't' for tone hpylm, 's' for shant hpylm
    cdef dict children # dictionary of deeper context HPYLNodes
    cdef str context # one previous element
    cdef dict tables # dict of all tables - key: segment, value: Tables
    cdef dict c_wh # c(w|h) .. count of all customers 
    cdef int c_h # sum of all counts in c(w|h)
    cdef int t_h # t_{h} .. sum of counts of tables in tables dictionary
    cdef bint add_tone(self, str tone, list context, float pwhcomma)
    cdef bint remove_tone(self, str tone, list context)
    cdef float get_pwh_probability(self, str tone, list context, float pwhcomma, float prev_pass_product)
    cdef int sample_context_length(self, str tone, list context, float pwhcomma)
    cdef void __fill_sample_prob_table(self, list sample_prob_table, str tone, list context, float pwhcomma, float prev_pass_product)

cdef class TTables():
    # t_hw could be get as len(self.tables)
    cdef list tables # list of ints - customer counts
    cdef str tone
    cdef THPYLMNode hpylmnode
    cdef bint add_customer(self, float pwhcomma, int t_h)
    cdef bint remove_customer(self)