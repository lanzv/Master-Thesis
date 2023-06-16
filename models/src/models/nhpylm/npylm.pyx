from chant cimport Chant, BOC, EOC, BOS, EOS
from random_utils cimport random_choice, poisson
import numpy as np
cimport numpy as np
from libc.math cimport log
np.import_array()
DTYPE = np.float64


cdef str EPSILON = "ε" # an empty character

cdef class NPYLM():
    def __cinit__(self, int max_segment_size, float init_d, float init_theta, 
                float init_a, float init_b, set tone_vocabulary, 
                float beta_stops, float beta_passes, float d_a, float d_b,
                float theta_alpha, float theta_beta):
        """
        Constructor of Nested Pitman-Yor Language Model.
        Initialize Segment Hierarchical Pitman-Yor Language Model.
            - current max depth
            - d and theta values for each depth
        Initialize Tone Hierarchical Pitman-Yor Language Model.
            - current max depth
            - d and theta values for each depth
            - beta stop/pass coefficients
            - init poisson a,b values
        Initialize all hyperparameters and NPYLM settings.

        Parameters
        ----------
        max_segment_size : int
            maximum segment size that the model considers
        init_d : float
            initial discount factor in G ~ PY(Gn, d, theta) - used in probability recursive computation
            it's used as a init value for each depth of both, thpylm and shpylm
            which is updated once the depth's values are learned
        init_theta : float
            initial theta that controls the average similarity of G and Gn in G ~ PY(Gn, d, theta) - used in probability recursive computation
            it's used as a init value for each depth of both, thpylm and shpylm
            which is updated once the depth's values are learned
        init_a : float
            initial 'a' argument in gamma distribution G(a, b) in order to estimate lambda for poisson correction
        init_b : float
            initial 'b' argument in gamma distribution G(a, b) in order to estimate lambda for poisson correction
        tone_vocabulary : set
            tone vocabulary set of all unique tones
        beta_stops : float
            stops hyperparameter, that will be used during sampling and probability computation in THPYLM
            that helps us avoid the situation of zero stops in depth that we want to include
        beta_passes : float
            stops hyperparameter, that will be used during sampling and probability computation in THPYLM
            that helps us to avoid the situation of zero passes in depth that we want to include
        d_a : float
            hyperparameter for the 'd' prediction, part of the alpha value of Beta prior
        d_b : float
            hyperparameter for the 'd' prediction, part of the beta value of Beta prior
        theta_alpha : float
            hyperparameter for the 'theta' prediction, part of the shape value of Gamma prior
        theta_beta : float
            hyperparameter for the 'theta' prediction, part of the scale value of Gamma prior
        """
        self.max_segment_size = max_segment_size
        self.init_d = init_d
        self.init_theta = init_theta
        self.d_a = d_a
        self.d_b = d_b
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.tone_vocabulary = tone_vocabulary
        self.root_tables_context_lengths = {}
        # Init SHPYLM settings
        self.shpylm_max_depth = -1
        self.shpylm_ds = []
        self.shpylm_thetas = []
        self.shpylm_root = SHPYLMNode(EPSILON, None, self, 0)
        # Init THPYLM settings
        self.thpylm_max_depth = -1
        self.thpylm_ds = []
        self.thpylm_thetas = []
        self.poisson_k_probs = [1.0/(max_segment_size+2) for _ in range(max_segment_size)]
        self.poisson_lambda = np.random.gamma(init_a, 1/init_b)
        self.beta_stops = beta_stops
        self.beta_passes = beta_passes
        self.init_poisson_a = init_a
        self.init_poisson_b = init_b
        self.thpylm_root = THPYLMNode(EPSILON, None, self, 0)

    cdef void add_chant(self, Chant chant):
        """
        Add chant segments to NPYLM.
        Add all segments from chant into SHPYLM with the bigram (one previous segment) context.
        In case of the newly created table in the root level of SHPYLM, add the segment into 
        THPYLM.

        Parameters
        ----------
        chant : Chant
            chant that contains actual segmentation we want to add to NPYLM
        """
        # First, add chant into segment HPYLM
        cdef str first_gram, second_gram
        cdef bint added_table
        cdef float pwhcomma

        pwhcomma = self.get_G0_probability(chant.segmentation[0])
        added_table = self.shpylm_root.add_segment(chant.segmentation[0], [BOC], pwhcomma) # First bigram
        if added_table:
            self.add_segment_to_thpylm(chant.segmentation[0])

        for first_gram, second_gram in zip(chant.segmentation[:-1], chant.segmentation[1:]): # All middle bigrams
            pwhcomma = self.get_G0_probability(second_gram)
            added_table = self.shpylm_root.add_segment(second_gram, [first_gram], pwhcomma)
            if added_table:
                self.add_segment_to_thpylm(second_gram)

        pwhcomma = self.get_G0_probability(EOC)
        added_table = self.shpylm_root.add_segment(EOC, [chant.segmentation[-1]], pwhcomma) # Last bigram
        if added_table:
            self.thpylm_root.add_tone(EOC, [], 1.0/len(self.tone_vocabulary))

    cdef void remove_chant(self, Chant chant):
        """
        Remove chant segments from NPYLM.
        Remove all segments from chant from SHPYLM with the bigram (one previous segment) context.
        In case of removed table in the root level of SHPYLM, remove also the segment from THPYLM.

        Parameters
        ----------
        chant : Chant
            chant that contains actual segmentation we want to remove from NPYLM
        """
        # First, remove chant from segment HPYLM
        cdef str first_gram, second_gram
        cdef bint removed_table

        removed_table = self.shpylm_root.remove_segment(chant.segmentation[0], [BOC]) # First bigram
        if removed_table:
            self.remove_segment_from_thpylm(chant.segmentation[0])

        for first_gram, second_gram in zip(chant.segmentation[:-1], chant.segmentation[1:]):
            removed_table = self.shpylm_root.remove_segment(second_gram, [first_gram])
            if removed_table:
                self.remove_segment_from_thpylm(second_gram)

        removed_table = self.shpylm_root.remove_segment(EOC, [chant.segmentation[-1]]) # Last bigram
        if removed_table:
            self.thpylm_root.remove_tone(EOC, [])


    cdef void add_segment_to_thpylm(self, str segment):
        """
        Add segment to THPYLM.
        Sample context length for all segment tones and add the each tone with its sampled context length
        (number of previous tones, we are sampling n for tone n-gram) into THPYLM.
        All generated context lengths are stored at the same table index as index of the new table in SHPYLM 
        tables corresponding to the segment, that was created - which was the reason of invoking this method.

        Parameters
        ----------
        segment : str
            segment as a string, each tone or special sign needs to be stored as a single character
            (!!<boc> or <eoc> are always 5 special signs/tones long segments, not one!!)
        """
        cdef list segment_tones = [BOS]+[*segment]+[EOS]
        cdef int i, ngram
        cdef list context_lengths = []
        cdef THPYLMNode thpylm_root = self.thpylm_root
        for i in range(0, len(segment_tones)):
            ngram = self.thpylm_root.sample_context_length(segment_tones[i], segment_tones[:i], 
                                                            1.0/len(self.tone_vocabulary))
            context_lengths.append(ngram)
            thpylm_root.add_tone(segment_tones[i], segment_tones[i-ngram:i], 1.0/len(self.tone_vocabulary))

        # Store the mapping of sampled context lengths and tables in SHPYLM
        if segment in self.root_tables_context_lengths:
            self.root_tables_context_lengths[segment].append(context_lengths)
        else:
            self.root_tables_context_lengths[segment] = [context_lengths]


    cdef void remove_segment_from_thpylm(self, str segment):
        """
        Remove segment from THPYLM.
        Remove all segment tones from THPYLM using the context lengths information that was stored based on the
        last table index that was removed. The root_tables_context_lengths[segment] was generated in the same order
        as the SHPYLM.tables[segment].tables, so indices should match. Therefore use the same context lengths to 
        remove the tone with the specific context from THPYLM (from the specific tree depth).
        Finally, remove the context lengths from root_tables_context_lengths[segment] since the corresponding table
        doesn't exist anymore, to keep the same order of both arrays in the same way.

        Parameters
        ----------
        segment : str
            segment as a string, each tone or special sign needs to be stored as a single character
            (!!<boc> or <eoc> are always 5 special signs/tones long segments, not one!!)
        """
        cdef list segment_tones = [BOS]+[*segment]+[EOS]
        cdef int i, ngram
        cdef int k_index = self.last_removed_shpylm_table_index
        for i, ngram in zip(range(0, len(segment_tones)), self.root_tables_context_lengths[segment][k_index]):
            self.thpylm_root.remove_tone(segment_tones[i], segment_tones[i-ngram:i])
        
        # Remove the context length set from the mapping
        del self.root_tables_context_lengths[segment][k_index]
        



    cdef float get_G0_probability(self, str segment):
        """
        When segment is EOC, just return 1/|V|.
        Otherwise compute G0 probability of given segment.
        The result probability is a product of all tone probabilities based on the
        full previous context that is provided in comming 'segment' string.
        As a init p(w|h') is used the 1/V, where V is the size of tone vocabulary.

        In case that segment is in the correct size range (based on provided max_segment_size),
        - which tells us that we are not in the "init thpylm" stage - apply poisson correction
        p(c1...ck) = (p(c1...ck, k | TPYLM)/p(k|TPYLM)) * Po(k, lambda)
        where k is the length of the segment and p(k|TPYLM) tells us a probability of segment to be
        size of k. Generally we would like to keep segments arround the average segment length, to avoid
        the one tone segments that should be more probable.

        Parameters
        ----------
        segment : str
            segment as a string, each tone or special sign needs to be stored as a single character
        Returns
        -------
        prob : float
            G0 probability, the probability with a lower context than the SHPYLM root - probability of
            segment 's' given the SHPYLM tree
        """
        if segment == EOC:
            return 1.0/len(self.tone_vocabulary)
        # First, get prob from tone HPYLM
        cdef float prob = 1.0
        cdef list segment_tones = [BOS] + [*segment] + [EOS]
        for i in range(1, len(segment_tones)):
            prob *= self.thpylm_root.get_pwh_probability(segment_tones[i], segment_tones[:i], 1.0/len(self.tone_vocabulary), 1.0)

        # Second, apply Poisson correction
        cdef int k = len(segment)
        if k <= self.max_segment_size: # apply poisson only in case of segments to be in the correct size
            prob = (prob/self.poisson_k_probs[k-1])*poisson(k,self.poisson_lambda)
        return prob

    cdef float get_bigram_probability(self, str first_gram, str second_gram):
        """
        Compute bigram probability p(second_gram | first_gram) regarding the Pitman Yor Language Model.
        First compute G0, then use it to feed the SHPYLM to compute probability from G1 and G2.  

        Parameters
        ----------
        first_gram : string
            first segment as a string, first gram
        second_gram : string
            second segment as a string when the first gram is given
        Returns
        -------
        prob : float
            bigram probability p(second_gram | first_gram)
        """
        cdef float prob = self.get_G0_probability(second_gram)
        
        # Get prob from segment HPYLM
        return self.shpylm_root.get_pwh_probability(second_gram, [first_gram], prob)
    

    cdef float get_segmentation_log_probability(self, list chant_segmentation):
        """
        Compute probability of the given chant segmentation (product of all bigrams).
        Apply logarithm on it.
        And return it back.

        Parameters
        ----------
        chant_segmentation : list of strings
            list of segments of chant we want to compute its log probability
        Returns
        -------
        prob_log : float
            logarithm probability of chant segmentation given NPYLM
        """
        cdef str first_gram, second_gram
        cdef float prob_log = 0.0
        prob_log += log(self.get_bigram_probability(BOC, chant_segmentation[0]))
        for first_gram, second_gram in zip(chant_segmentation[:-1], chant_segmentation[1:]):
            prob_log += log(self.get_bigram_probability(first_gram, second_gram))
        prob_log += log(self.get_bigram_probability(chant_segmentation[-1], EOC))
        return prob_log












# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX          Segment Hierarchical Pitman-Yor Tree Node          XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX





"""
Segment Hierarchilcal Pitman-Yor Language Model Node

SHPYLM is represented as a tree, starting with in the root with parent = null
Each node has information about actual 
    context ... prvevious segments for instance "aaa"(root) -> "bbb"(1. depth) -> "ccc"(2.depth), 
        segments are in the chant stored as "ccc bbb aaa segment" (=4gram example p(segment | ccc bbb aaa)).
    tables ... dictionary of restaurants - list of tables for the specific segment (stored in STables data structure)
    c_wh ... dictionary of customer counts - sum of all custmer along all tables for the specific segment
    t_h ... number of all tables over all segments, only in this node (not considering parents/children)
    c_h ... sum of all numbers in c_wh over all segments
"""
cdef class SHPYLMNode():
    def __cinit__(self, str context, SHPYLMNode parent, NPYLM npylm, int depth):
        """
        Initialize the SHPYLM node.
        Extend lists of npylm d and theta hyperparameters if needed.

        Parameters
        ----------
        context : str
            segment that is predecessor of parent's contexts which are predecessors of current segment
        parent : SHPYLMNode
            SHPYLM parent node, or null if we are in the root node
        npylm : NPYLM
            NPYLM object that has information of all current model settings.
        depth : int
            depth of shpylm tree, root is 0
        """
        self.parent = parent
        self.children = {}
        self.context = context
        self.tables = {}
        self.c_wh = {}
        self.c_h = 0
        self.t_h = 0
        self.npylm = npylm
        self.depth = depth
        if depth > npylm.shpylm_max_depth:
            npylm.shpylm_max_depth = depth
                    
        if depth >= len(npylm.shpylm_ds) and depth >= len(npylm.shpylm_thetas):
            npylm.shpylm_ds.append(npylm.init_d)
            npylm.shpylm_thetas.append(npylm.init_theta)
    
    cdef bint add_segment(self, str segment, list context, float pwhcomma):
        """
        1. Compute p(w|h) of this context depending on the previous pwhcomma
        p(w|h) = ((c(w|h)-d*t_hw)/(theta + c(h))) + ((theta + d*t_h)/(theta+c(h)))*p(w|h') 
        2. Propagate the p(w|h), segment we want to add and list of the not placed context
        3. If we are in the depth we want to (all context segments are placed), add customer to restaurant
            we need p(w|h) for that ... if new tables will be generated (based on the sampling), 
            In case that new table was generated in the children's node, we also add customer
            to the restaurant of this node (for the table on the level below).

        Parameters
        ----------
        segment : str
            segment we want to add into SHPYLM
        context : list of strings
            list of not placed context (previous) segments
            the root takes all previous context segments, its children takes only context list without the last
            segment (the one most closest to the current segment we are adding) - those closer are already stored
            in parent nodes
        pwhcomma : float
            p(w|h'), probability with the smaller context
        Returns
        -------
        new_table : bint
            boolean that says whether the new table was created in this node or not
        """
        cdef bint new_table = False
        cdef bint new_prev_table = True
        cdef STables segment_restaurant
        cdef SHPYLMNode context_child
        cdef float pwh
        cdef float d = self.npylm.shpylm_ds[self.depth]
        cdef float theta = self.npylm.shpylm_thetas[self.depth]

        # Compute p(w|h)
        if segment in self.tables:
            segment_restaurant = self.tables[segment]
            pwh = ((self.c_wh[segment] - d*len(segment_restaurant.tables))/(theta + self.c_h)) \
                        + (((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma)
        else:
            # For not existing tone in tables, use only the second part of equation
            pwh = ((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma


        # Propagate segments to the next level
        if len(context) > 0:
            if not context[-1] in self.children:
                context_child = SHPYLMNode(context[-1], self, self.npylm, self.depth + 1)
                self.children[context[-1]] = context_child
            else:
                context_child =  self.children[context[-1]]
            new_prev_table = context_child.add_segment(segment, context[:-1], pwh)


        # If new table was generated in the lower level, add the customer here
        if new_prev_table:
            if segment in self.c_wh and segment in self.tables:
                self.c_wh[segment] += 1
                segment_restaurant = self.tables[segment]
            else:
                self.c_wh[segment] = 1
                segment_restaurant = STables(segment, self)
                self.tables[segment] = segment_restaurant
            self.c_h += 1
            new_table = segment_restaurant.add_customer(pwhcomma, self.t_h)
            if new_table:
                self.t_h += 1

        return new_table
        



    cdef bint remove_segment(self, str segment, list context):
        """
        1. Propagate segment we want to remove and list of the not searched context
        2. If we are in the depth we want to (all context segments are searched/used), remove customer 
            from segment's restaurant ... 
            if there is a table that got empty (we have to remove it based on the sampling), 
            in the children's node, we also remove customer from the restuarant of this node specific for the segment

        Parameters
        ----------
        segment : str
            segment we want to remove from SHPYLM
        context : list of strings
            list of not used context (previous) segments for finding the node where we need to remove our segment
        Returns
        -------
        removed_table : bint
            boolean that says whether there is a table which was removed in this node or not
        """
        cdef SHPYLMNode context_child
        cdef bint removed_table = False
        cdef bint removed_prev_table = True
        cdef STables segment_restaurant
        cdef float d = self.npylm.shpylm_ds[self.depth]

        # Propagate removing to the next level
        if len(context) > 0:
            context_child = self.children[context[-1]]
            removed_prev_table = context_child.remove_segment(segment, context[:-1])
            if context_child.t_h == 0:
                self.children.pop(context[-1])


        # Remove customer from table on this level
        if removed_prev_table:
            self.c_wh[segment] -= 1
            self.c_h -= 1
            segment_restaurant = self.tables[segment]
            removed_table = segment_restaurant.remove_customer()
            if self.c_wh[segment] == 0 or len(segment_restaurant.tables) == 0:
                self.c_wh.pop(segment)
                self.tables.pop(segment)
            if removed_table:
                self.t_h -= 1

        return removed_table


    cdef float get_pwh_probability(self, str segment, list context, float pwhcomma):
        """
        1. Compute p(w|h) of this context depending on the previous pwhcomma
        p(w|h) = ((c(w|h)-d*t_hw)/(theta + c(h))) + ((theta + d*t_h)/(theta+c(h)))*p(w|h') 
        2. Propagate the p(w|h) till the there is no more context we are interested in
            and return back the final probability value


        Parameters
        ----------
        segment : str
            segment we want to add into SHPYLM
        context : list of strings
            list of not searched/used context (previous) segments
        pwhcomma : float
            p(w|h'), probability with the smaller context
        Returns
        -------
        prob : float
            the p(w|h) probability 
        """
        cdef float prob = pwhcomma
        cdef SHPYLMNode context_child
        cdef STables seg_tables
        cdef float d = self.npylm.shpylm_ds[self.depth]
        cdef float theta = self.npylm.shpylm_thetas[self.depth]
        if segment in self.tables:
            seg_tables = self.tables[segment]
            prob = ((self.c_wh[segment] - d*len(seg_tables.tables))/(theta + self.c_h)) \
                        + (((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma)
        else:
            prob = ((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma
        if len(context) > 0 and context[-1] in self.children:
            context_child = self.children[context[-1]]
            prob = context_child.get_pwh_probability(segment, context[:-1], prob)
        return prob


"""
Restaurant for the specific segment - Segment Tables
There are tables and each table has several seats (!non zero!).
List tables contains numbers of seats at table of index k.
"""
cdef class STables():
    def __cinit__(self, str segment, SHPYLMNode hpylmnode):
        """
        Init STables - Segment Tables - Segment Restaurant

        Parameters
        ----------
        segment : str
            the restaurant is created for only this segment at specific node (based on context)
        hpylmnode : SHPYLMNode
            the SHPYLM Node that owns this restaurant
        """
        self.tables = []
        self.hpylmnode = hpylmnode
        self.segment = segment

    
    cdef bint add_customer(self, float pwhcomma, int t_h):
        """
        Add customer into restaurant.
        Use the p(w|h') and choose the table for the customer with prob distribution
        (c_hwk-d) for kth table (c_whk is a count of customers with segment w, context h, that are sitting at table k)
        (theta+(d*t_h))*p(w|h') for the creating new table

        Parameters
        ----------
        pwhcomma : float
            p(w|h'), probability of segment with shorter context
        t_h : int
            number of all tables over all restaurant in the w|h node
            (node described by context h and segment w)
        Returns
        -------
        new_table : bint
            boolean that says whether the new table was created or not
        """
        cdef int table_k
        cdef int c_hwk # customer count at table 'k'
        cdef list posibilities = []
        cdef bint new_table = False
        cdef float d = self.hpylmnode.npylm.shpylm_ds[self.hpylmnode.depth]
        cdef float theta = self.hpylmnode.npylm.shpylm_thetas[self.hpylmnode.depth]

        for c_hwk in self.tables:
            posibilities.append(max(0, c_hwk - d))
        if len(posibilities) > 0:
            posibilities.append((theta+(d*t_h))*pwhcomma)
            table_k = random_choice(posibilities)
        else:
            table_k = 0 # create new table in the next step

        if table_k == len(self.tables):
            self.tables.append(1)
            new_table = True
        else:
            self.tables[table_k] += 1

        return new_table
    
    cdef bint remove_customer(self):
        """
        Remove customer from restaurant.
        Choose the table with probability distribution
        (c_hwk) for kth table (c_whk is a count of customers with segment w, context h, that are sitting at table k)
        and remove the customer from it.

        If the table gets empty, remove it.
        And store the information of removed table index into NPYLM last_removed_shpylm_table_index that will be used 
        removing from THPYLM

        Returns
        -------
        removed_table : bint
            boolean that says whether there is a removed table
        """
        cdef int table_k
        cdef int c_hwk # customer count at table 'k'
        cdef list posibilities = []
        cdef bint removed_table = False

        for c_hwk in self.tables:
            posibilities.append(c_hwk) # during the adding process, the minus discount factor is included
        table_k = random_choice(posibilities)
        self.hpylmnode.npylm.last_removed_shpylm_table_index = table_k

        self.tables[table_k] -= 1
        if self.tables[table_k] == 0:
            del self.tables[table_k]
            removed_table = True

        return removed_table













# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX           Tone Hierarchical Pitman-Yor Tree Node           XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
cdef thpylm_trehshold = 1e-12







"""
Tone Hierarchilcal Pitman-Yor Language Model Node

THPYLM is represented as a tree, starting with in the root with parent = null
Each node has information about current 
    context ... for backtracking prvevious tones the same way as in the THPYLM
    tables ... dictionary of restaurants - list of tables for the specific tones (stored in STables data structure)
    c_wh ... dictionary of customer counts - sum of all custmer along all tables for the specific tone
    t_h ... number of all tables over all tones, only in this node (not considering parents/children)
    c_h ... sum of all numbers in c_wh over all tones
"""
cdef class THPYLMNode():
    def __cinit__(self, str context, THPYLMNode parent, NPYLM npylm, int depth):
        """
        Initialize the THPYLM node.
        Extend lists of npylm d and theta hyperparameters if needed.

        Parameters
        ----------
        context : str
            tone that is predecessor of parent's contexts which are predecessors of current tone
        parent : THPYLMNode
            THPYLM parent node, or null if we are in the root node
        npylm : NPYLM
            NPYLM object that has information of all current model settings.
        depth : int
            depth of thpylm tree, root is 0
        """
        self.parent = parent
        self.children = {}
        self.context = context
        self.tables = {}
        self.c_wh = {}
        self.c_h = 0
        self.t_h = 0
        self.npylm = npylm
        self.stops = 0
        self.passes = 0
        self.depth = depth
        if depth > npylm.thpylm_max_depth:
            npylm.thpylm_max_depth = depth
            
        if depth >= len(npylm.thpylm_ds) and depth >= len(npylm.thpylm_thetas):
            npylm.thpylm_ds.append(npylm.init_d)
            npylm.thpylm_thetas.append(npylm.init_theta)
            

    cdef bint add_tone(self, str tone, list context, float pwhcomma):
        """
        Compute p(w|h) = ((c(w|h)-d*t_hw)/(theta + c(h))) + ((theta + d*t_h)/(theta+c(h)))*p(w|h'),
        propagate p(w|h) to the deeper nodes with higher context, till the context list is not empty.
        Then add the customer to the node. If the new table was created, add the new customer also to the
        parent node (and recursivaly the same about parent's parent).

        Parameters
        ----------
        tone : str
            tone we want to add into THPYLM
        context : list of strings
            list of not placed context (previous) tones
        pwhcomma : float
            p(w|h'), probability of tone w with the smaller context
        Returns
        -------
        new_table : bint
            boolean that says whether the new table was created in this node or not
        """
        cdef bint new_table = False
        cdef bint new_prev_table = False
        cdef TTables tone_restaurant
        cdef THPYLMNode context_child
        cdef float pwh
        cdef THPYLMNode parent
        cdef float d = self.npylm.thpylm_ds[self.depth]
        cdef float theta = self.npylm.thpylm_thetas[self.depth]

        # Compute p(w|h)
        if tone in self.tables:
            tone_restaurant = self.tables[tone]
            pwh = ((self.c_wh[tone] - d*len(tone_restaurant.tables))/(theta + self.c_h)) \
                        + (((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma)
        else:
            # For not existing tone in tables, use only the second part of equation
            pwh = ((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma


        # Propagate segments to the next level
        if len(context) > 0:
            if not context[-1] in self.children:
                context_child = THPYLMNode(context[-1], self, self.npylm, self.depth+1)
                self.children[context[-1]] = context_child
            else:
                context_child =  self.children[context[-1]]
            new_prev_table = context_child.add_tone(tone, context[:-1], pwh)
        else:
            if tone in self.c_wh and tone in self.tables:
                self.c_wh[tone] += 1
                tone_restaurant = self.tables[tone]
            else:
                self.c_wh[tone] = 1
                tone_restaurant = TTables(tone, self)
                self.tables[tone] = tone_restaurant
            self.c_h += 1
            new_table = tone_restaurant.add_customer(pwhcomma, self.t_h)
            if new_table:
                self.t_h += 1
            # update stops and passes
            self.stops += 1
            parent = self.parent
            while parent != None:
                parent.passes += 1
                parent = parent.parent


        # If new table was generated in the lower level, add the customer here
        if new_prev_table:
            if tone in self.c_wh and tone in self.tables:
                self.c_wh[tone] += 1
                tone_restaurant = self.tables[tone]
            else:
                self.c_wh[tone] = 1
                tone_restaurant = TTables(tone, self)
                self.tables[tone] = tone_restaurant
            self.c_h += 1
            new_table = tone_restaurant.add_customer(pwhcomma, self.t_h)
            if new_table:
                self.t_h += 1

        return new_table
        



    cdef bint remove_tone(self, str tone, list context):
        """
        1. Propagate tone we want to remove and list of the not used/searched context yet
        2. If we are in the depth we want to (all context tones are searched/used), remove customer 
            from tone's restaurant ... 
            if there is a table that got empty (we have to remove it based on the sampling), 
            in the children's node, we also remove customer from the restuarant of this node specific for the tone

        Parameters
        ----------
        tone : str
            tone we want to remove from THPYLM
        context : list of strings
            list of not used context (previous) tones for finding the node where we need to remove our tone
        Returns
        -------
        removed_table : bint
            boolean that says whether there is a table which was removed in this node or not
        """
        cdef THPYLMNode context_child
        cdef bint removed_table = False
        cdef bint removed_prev_table = False
        cdef TTables tone_restaurant

        # Propagate removing to the next level
        if len(context) > 0:
            context_child = self.children[context[-1]]
            removed_prev_table = context_child.remove_tone(tone, context[:-1])
            if context_child.t_h == 0:
                self.children.pop(context[-1])
        else:
            self.c_wh[tone] -= 1
            self.c_h -= 1
            tone_restaurant = self.tables[tone]
            removed_table = tone_restaurant.remove_customer()
            if self.c_wh[tone] == 0 or len(tone_restaurant.tables) == 0:
                self.c_wh.pop(tone)
                self.tables.pop(tone)
            if removed_table:
                self.t_h -= 1
            # update stops and passes
            self.stops -= 1
            parent = self.parent
            while parent != None:
                parent.passes -= 1
                parent = parent.parent

        # Remove customer from table on this level
        if removed_prev_table:
            self.c_wh[tone] -= 1
            self.c_h -= 1
            tone_restaurant = self.tables[tone]
            removed_table = tone_restaurant.remove_customer()
            if self.c_wh[tone] == 0 or len(tone_restaurant.tables) == 0:
                self.c_wh.pop(tone)
                self.tables.pop(tone)
            if removed_table:
                self.t_h -= 1

        return removed_table


    cdef float get_pwh_probability(self, str tone, list context, float pwhcomma, float prev_pass_product):
        """
        1. Compute stop probability (that this node is the one that is more often final context tone (final context length) than
        the middle context tone) as 
        stop_prob = (node_stop_counts+beta_stops)/(node_stop_counts+beta_stops+node_pass_counts+beta_passes)
        2. Compute pass probability the same way as 
        pass_prob = (node_pass_counts+beta_passes)/(node_stop_counts+beta_stops+node_pass_counts+beta_passes)
        
        3. Compute p(w|h) of this context depending on the previous pwhcomma
        p(w|h) = ((c(w|h)-d*t_hw)/(theta + c(h))) + ((theta + d*t_h)/(theta+c(h)))*p(w|h')

        4. Propagate the p(w|h) and prev_pass_prob_product*pass_prob till the there is no more context we are interested in
            and return back the final probability value
        5. sum up the new deeper probability (with longer context) with the current probability as
            prob = longer_context_prob + current_context_prob*stop_prob*prev_pass_prob_product
            where prev_pass_prob_product is a product of all previous node's pass probabilities (where the current one is not 
            used in this product)
            and return the prob to the previous node

        We consider infinite ngrams, so there is a threshold to end up the process earlier. If there is no context anymore, we can 
        still consider those deeper ngrams as well.

        Parameters
        ----------
        tone : str
            tone we want to add into THPYLM
        context : list of strings
            list of not searched/used context (previous) tones
        pwhcomma : float
            p(w|h'), probability with the smaller context
        prev_pass_product : float
            product of all previous node's pass probabilities
        Returns
        -------
        prob : float
            probability of tone w with the context h
        """
        cdef float pass_prob
        cdef float stop_prob
        cdef float prob = pwhcomma
        cdef float next_prob
        cdef float pwh
        cdef THPYLMNode context_child
        cdef TTables seg_tables
        cdef float new_stop_prob
        cdef float d = self.npylm.thpylm_ds[self.depth]
        cdef float theta = self.npylm.thpylm_thetas[self.depth]
        stop_prob = (self.stops + self.npylm.beta_stops)/\
                (self.stops + self.npylm.beta_stops + self.passes + self.npylm.beta_passes)
        pass_prob = (self.passes + self.npylm.beta_passes)/\
                (self.stops + self.npylm.beta_stops + self.passes + self.npylm.beta_passes)

        # If the threshold condition is not fulfiled, return 0.0
        if thpylm_trehshold < stop_prob * prev_pass_product:
            # Compute current p(w|h), then get the next p(w|h) with bigger context, 
            # then apply stop and pass probabilities into equation and send it to the
            # previous layer.
            if tone in self.tables:
                seg_tables = self.tables[tone]
                pwh = ((self.c_wh[tone] - d*len(seg_tables.tables))/(theta + self.c_h)) \
                            + (((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma)
            else:
                # For not existing tone in tables, use only the second part of equation
                pwh = ((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma
            if len(context) > 0 and context[-1] in self.children:
                context_child = self.children[context[-1]]
                next_prob = context_child.get_pwh_probability(tone, context[:-1], pwh, prev_pass_product*pass_prob)
                prob = next_prob + pwh*stop_prob*prev_pass_product
            else:
                # In the case of not existing context in tree, use the Beta priors
                prob = 0
                new_stop_prob = (self.npylm.beta_stops/(self.npylm.beta_stops + self.npylm.beta_passes))
                while thpylm_trehshold < new_stop_prob * prev_pass_product:
                    prob += new_stop_prob * prev_pass_product * pwh
                    prev_pass_product *= (self.npylm.beta_passes/(self.npylm.beta_stops + self.npylm.beta_passes))    
        else:
            prob = 0
        return prob


    cdef int sample_context_length(self, str tone, list context, float pwhcomma):
        """
        Sample length (in range of 0 and len(context)) of context that should be added into THPYLM.

        Parameters
        ----------
        tone : str
            tone we want to add into THPYLM
        context : list of strings
            list of not searched/used context (previous) tones
        pwhcomma : float
            p(w|h'), probability with the smaller context
        Returns
        -------
        sampled_length : int
            context length that was sampled based on conditional probability distribution for the specific tone and its context
        """
        if len(context) == 0:
            return 0
        cdef list sample_prob_table = []
        self.__fill_sample_prob_table(sample_prob_table, tone, context, pwhcomma, 1)
        return random_choice(sample_prob_table)


    cdef void __fill_sample_prob_table(self, list sample_prob_table, str tone, list context, 
                                        float pwhcomma, float prev_pass_product):
        """
        Fill the sample_prob_table - table of probabilities of all possible context lenghts
        For the one context length compute the probability as
        1. Compute stop probability (that this node is the one that is more often final context tone (final context length) than
        the middle context tone) as 
        stop_prob = (node_stop_counts+beta_stops)/(node_stop_counts+beta_stops+node_pass_counts+beta_passes)
        2. Compute pass probability the same way as 
        pass_prob = (node_pass_counts+beta_passes)/(node_stop_counts+beta_stops+node_pass_counts+beta_passes)
        
        3. Compute p(w|h) of this context depending on the previous pwhcomma
        p(w|h) = ((c(w|h)-d*t_hw)/(theta + c(h))) + ((theta + d*t_h)/(theta+c(h)))*p(w|h')

        context_length_prob = pwh*stop_prob*prev_pass_product


        Parameters
        ----------
        sample_prob_table : list of floats
            list of probabilities for all possible context lengths that should be filled at the and of the recursive process
        tone : str
            tone we want to add into THPYLM
        context : list of strings
            list of not searched/used context (previous) tones
        pwhcomma : float
            p(w|h'), probability with the smaller context
        prev_pass_product : float
            product of all previous node's pass probabilities
        """
        cdef float pass_prob
        cdef float stop_prob
        cdef float pwh
        cdef THPYLMNode context_child
        cdef TTables seg_tables
        cdef float new_stop_prob
        cdef float d = self.npylm.thpylm_ds[self.depth]
        cdef float theta = self.npylm.thpylm_thetas[self.depth]

        stop_prob = (self.stops + self.npylm.beta_stops)/\
                (self.stops + self.npylm.beta_stops + self.passes + self.npylm.beta_passes)
        pass_prob = (self.passes + self.npylm.beta_passes)/\
                (self.stops + self.npylm.beta_stops + self.passes + self.npylm.beta_passes)
        # Compute current p(w|h), then get the next p(w|h) with bigger context, 
        # then apply stop and pass probabilities into equation and send it to the
        # previous layer.
        if tone in self.tables:
            seg_tables = self.tables[tone]
            pwh = ((self.c_wh[tone] - d*len(seg_tables.tables))/(theta + self.c_h)) \
                        + (((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma)
        else:
            # For not existing tone in tables, use only the second part of equation
            pwh = ((theta + d*self.t_h)/(theta + self.c_h)) * pwhcomma
        sample_prob_table.append(pwh*stop_prob*prev_pass_product)

        if len(context) > 0 and context[-1] in self.children:
            context_child = self.children[context[-1]]
            context_child.__fill_sample_prob_table(sample_prob_table, tone, context[:-1], pwh, prev_pass_product*pass_prob)
        else:
            # In the case of not existing context in tree, use the Beta priors
            new_stop_prob = (self.npylm.beta_stops/(self.npylm.beta_stops + self.npylm.beta_passes))
            prev_pass_product *= pass_prob
            for _ in context:
                sample_prob_table.append(new_stop_prob * prev_pass_product * pwh)
                prev_pass_product *= (self.npylm.beta_passes/(self.npylm.beta_stops + self.npylm.beta_passes))    
    



"""
Restaurant for the specific tone - Tone Tables 
(it differs a little bit from STables so it's better to keep them separated
due to avoid some unnecessary)
There are tables and each table has several seats (!non zero!).
List tables contains numbers of seats at table of index k.
"""
cdef class TTables():
    def __cinit__(self, str tone, THPYLMNode hpylmnode):
        """
        Init TTables - Tone Tables - Tone Restaurant

        Parameters
        ----------
        tone : str
            the restaurant is created for only this tone at specific node (based on context)
        hpylmnode : THPYLMNode
            the THPYLM Node that owns this restaurant
        """
        self.tables = []
        self.hpylmnode = hpylmnode
        self.tone = tone

    
    cdef bint add_customer(self, float pwhcomma, int t_h):
        """
        Add customer into restaurant.
        Use the p(w|h') and choose the table for the customer with prob distribution
        (c_hwk-d) for kth table (c_whk is a count of customers with tone w, context h, that are sitting at table k)
        (theta+(d*t_h))*p(w|h') for the creating new table

        Parameters
        ----------
        pwhcomma : float
            p(w|h'), probability of tone with shorter context
        t_h : int
            number of all tables over all restaurant in the w|h node
            (node described by context h and tone w)
        Returns
        -------
        new_table : bint
            boolean that says whether the new table was created or not
        """
        cdef int table_k
        cdef int c_hwk # customer count at table 'k'
        cdef list posibilities = []
        cdef bint new_table = False
        cdef float d = self.hpylmnode.npylm.thpylm_ds[self.hpylmnode.depth]
        cdef float theta = self.hpylmnode.npylm.thpylm_thetas[self.hpylmnode.depth]

        for c_hwk in self.tables:
            posibilities.append(max(0, c_hwk - d))
        if len(posibilities) > 0:
            posibilities.append((theta+(d*t_h))*pwhcomma)
            table_k = random_choice(posibilities)
        else:
            table_k = 0 # create new table in the next step

        if table_k == len(self.tables):
            self.tables.append(1)
            new_table = True
        else:
            self.tables[table_k] += 1

        return new_table
    
    cdef bint remove_customer(self):
        """
        Remove customer from restaurant.
        Choose the table with probability distribution
        (c_hwk) for kth table (c_whk is a count of customers with tone w, context h, that are sitting at table k)
        and remove the customer from it.

        If the table gets empty, remove it.

        Returns
        -------
        removed_table : bint
            boolean that says whether there is a removed table
        """
        cdef int table_k
        cdef int c_hwk # customer count at table 'k'
        cdef list posibilities = []
        cdef bint removed_table = False

        for c_hwk in self.tables:
            posibilities.append(c_hwk) # during the adding process, the minus discount factor is included
        table_k = random_choice(posibilities)

        self.tables[table_k] -= 1
        if self.tables[table_k] == 0:
            del self.tables[table_k]
            removed_table = True

        return removed_table