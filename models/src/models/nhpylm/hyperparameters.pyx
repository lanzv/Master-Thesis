from npylm cimport NPYLM, SHPYLMNode, STables, THPYLMNode, TTables
from chant cimport Chant, BOS, EOS
from random_utils cimport random_choice
import numpy as np
cimport numpy as np
from libc.math cimport log
np.import_array()
DTYPE = np.float64

cdef void apply_hyperparameters_learning(NPYLM npylm, list train_chants):
    """
    Learn 
        - theta of all depths for both SHPYLM, THPYLM
        - probability of all possible segment lengths
        - lambda value for poisson correction
    
    Parameters
    ----------
    npylm : NPYLM
        npylm wich we want to update and learn hyperparameters 
    train_chants : list of Chants
        list of training chants represented as Chants, mainly for the poisson lambda calculation
    """
    update_poisson_lambda(npylm, train_chants)
    update_poisson_k_probs(npylm)
    update_shpylm_d_theta(npylm)
    update_thpylm_d_theta(npylm)

cdef void update_poisson_lambda(NPYLM npylm, list train_chants):
    """
    Update the lambda value for poisson correction using the Gamma distribution.
    new_lambda = Gamma(a, b), where 
        a = Sum_{segments}(len(segment)*t_wh) of root
        b = (init_poisson_b + t_h) of root 
    
    Parameters
    ----------
    npylm : NPYLM
        npylm wich we want to update and learn hyperparameters 
    train_chants : list of Chants
        list of training chants represented as Chants
    """
    cdef float a = npylm.init_poisson_a
    cdef float b = npylm.init_poisson_b + npylm.shpylm_root.t_h
    cdef str segment
    cdef int segment_length
    cdef int tables_count
    cdef Chant chant
    cdef set used_segments = set()
    cdef SHPYLMNode shpylm_root = npylm.shpylm_root
    cdef STables segment_restaurant

    # Get a, b from all possible unique segments
    for chant in train_chants:
        for segment in chant.segmentation:
            segment_length = len(segment)
            if (segment_length <= npylm.max_segment_size) and (not segment in used_segments):
                used_segments.add(segment)
                segment_restaurant = shpylm_root.tables[segment]
                tables_count = len(segment_restaurant.tables)
                a += (tables_count * segment_length)

    # Sample new lambda from a,b
    npylm.poisson_lambda = np.random.gamma(a, 1/b)


cdef void update_poisson_k_probs(NPYLM npylm, int segment_samples = 20000):
    """
    Sample some number of random segments (by default 20000) regarding the distribution of
    THPYLM (for each new segment sample tones till you get EOS). Keep the statistics about their lengths. 
    Then update poisson_k_probs of npylm regarding the new statistics.
    
    Parameters
    ----------
    npylm : NPYLM
        npylm wich we want to update and learn hyperparameters 
    segment_samples : int
        number of segments that are randomly generated
    """
    cdef np.ndarray length_occurences = np.zeros([npylm.max_segment_size], dtype=DTYPE)
    cdef int i
    cdef list vocabulary = list(npylm.tone_vocabulary)
    cdef list context
    cdef list next_tone_probs
    cdef str tone
    cdef int tone_index
    cdef int final_segment_length
    cdef THPYLMNode thpylm_root = npylm.thpylm_root

    for _ in range(segment_samples):
        # sample random segment from THPYLM
        context = [BOS]
        next_tone_probs = []
        # find the first not-eos tone
        for tone in vocabulary:
            next_tone_probs.append(thpylm_root.get_pwh_probability(tone, context, 1.0/len(vocabulary), 1.0))
        tone = vocabulary[random_choice(next_tone_probs)]
        context.append(tone)
        # sample next tones till you find the EOS character
        while tone != EOS and len(context) < npylm.max_segment_size + 1: # first element is BOS
            next_tone_probs = []
            for tone in vocabulary + [EOS]:
                next_tone_probs.append(thpylm_root.get_pwh_probability(tone, context, 1.0/len(vocabulary), 1.0))
            tone_index = random_choice(next_tone_probs)
            # In case of EOS, don't add tone to context
            if tone_index == len(vocabulary):
                tone = EOS
            else:
                tone = vocabulary[tone_index]
                context.append(tone)
        # store the final k length into statistics
        final_segment_length = len(context) - 1
        length_occurences[final_segment_length-1] += 1



    for i in range(npylm.max_segment_size):
        # Apply Laplace smoothing
        npylm.poisson_k_probs[i] = (length_occurences[i] + 1)/(segment_samples + npylm.max_segment_size)





# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX          Segment Hierarchical Pitman-Yor Tree Hyperparameters          XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


cdef void update_shpylm_d_theta(NPYLM npylm):
    """
    Update d and theta arrays of SHPYLM in NPYLM.
    First recursivaly precompute sum(1-y_ui), sum(1-z_uwkj), sum(y_ui), sum(log(x_u))
    over all nodes for each depth separately.
    After that compute new 'd' using Beta priors as Beta(d_a+sum(1-y_ui), d_b+sum(1-z_uwkj))
        where d_a and d_b are defautly set hyperparameters
    And compute new 'theta' using Gamma priors as Gamma(theta_alpha+sum(u_i), 1/(theta_beta+sum(log(x_u))))
        where theta_alpha and theta_beta are defautly set hyperparameters
    for all shpylm tree depths.

    Parameters
    ----------
    npylm : NPYLM
        npylm wich we want to update and learn hyperparameters 
    """
    cdef int i
    cdef float alpha, beta
    # Init value arrays for d,theta updates
    cdef np.ndarray sum1_minus_y_ui = np.zeros([npylm.shpylm_max_depth+1], dtype=DTYPE)
    cdef np.ndarray sum1_minus_z_uwkj = np.zeros([npylm.shpylm_max_depth+1], dtype=DTYPE)
    cdef np.ndarray sumy_ui = np.zeros([npylm.shpylm_max_depth+1], dtype=DTYPE)
    cdef np.ndarray sumlogx_u = np.zeros([npylm.shpylm_max_depth+1], dtype=DTYPE)
    # Set the first depth
    sum1_minus_y_ui[0] = __get_shpylm_1_minus_y_ui(npylm.shpylm_root)
    sum1_minus_z_uwkj[0] = __get_shpylm_1_minus_z_uwkj_sum(npylm.shpylm_root)
    sumy_ui[0] = __get_shpylm_y_ui_sum(npylm.shpylm_root)
    sumlogx_u[0] = __get_shpylm_logx_u(npylm.shpylm_root)
    # Fill array values for all depths recursivaly
    __recursive_shpylm_d_theta_preparation(npylm.shpylm_root, sum1_minus_y_ui, sum1_minus_z_uwkj, sumy_ui, sumlogx_u)
    
    # Set new d, theta for all depths
    for i in range(npylm.shpylm_max_depth+1):
        # sample d
        alpha = npylm.d_a + sum1_minus_y_ui[i]
        beta = npylm.d_b + sum1_minus_z_uwkj[i]
        npylm.shpylm_ds[i] = np.random.beta(alpha, beta)
        # sample theta
        shape = npylm.theta_alpha + sumy_ui[i]
        scale = 1.0/(npylm.theta_beta - sumlogx_u[i])
        npylm.shpylm_thetas[i] = np.random.gamma(shape, scale)


cdef void __recursive_shpylm_d_theta_preparation(SHPYLMNode node, np.ndarray sum1_minus_y_ui, 
                                                np.ndarray sum1_minus_z_uwkj, np.ndarray sumy_ui,
                                                np.ndarray sumlogx_u):
    """
    Recursivaly precompute sum(1-y_ui), sum(1-z_uwkj), sum(y_ui), sum(log(x_u))
    over all nodes for each depth separately.

    Parameters
    ----------
    node : SHPYLMNode
        SHPYLM node of tree we want to add to our precomputed values
    sum1_minus_y_ui : np.ndarray
        array of floats of sum(1-y_ui) for all depths
    sum1_minus_z_uwkj : np.ndarray
        array of floats of sum(1-z_uwkj) for all depths
    sumy_ui : np.ndarray
        array of floats of sum(y_ui) for all depths
    sumlogx_u : np.ndarray
        array of floats of sum(log(x_u)) for all depths
    """
    cdef SHPYLMNode child
    cdef str context
    for context in node.children:
        child = node.children[context]
        # Update value arrays
        sum1_minus_y_ui[child.depth] += __get_shpylm_1_minus_y_ui(child)
        sum1_minus_z_uwkj[child.depth] += __get_shpylm_1_minus_z_uwkj_sum(node)
        sumy_ui[child.depth] += __get_shpylm_y_ui_sum(child)
        sumlogx_u[child.depth] += __get_shpylm_logx_u(child)
        # Recursive call
        __recursive_shpylm_d_theta_preparation(child, sum1_minus_y_ui, sum1_minus_z_uwkj, sumy_ui, sumlogx_u)


cdef float __get_shpylm_1_minus_y_ui(SHPYLMNode node):
    """
    Sample sum_{i=1..t_h}(1-bernoulli((theta)/(theta+d*i))) over all tables in the specific node.
    
    Parameters
    ----------
    node : SHPYLMNode
        SHPYLM node of tree we want to add to our precomputed values
    Returns
    -------
    final_sum : float
        sum_{i=1..t_h}(1-bernoulli((theta)/(theta+d*i)))
    """
    cdef float final_sum = 0.0
    cdef float prob
    cdef int i
    cdef NPYLM npylm = node.npylm
    if node.t_h >= 2:
        for i in range(1,node.t_h):
            prob = npylm.shpylm_thetas[node.depth] / \
                    (npylm.shpylm_thetas[node.depth] + npylm.shpylm_ds[node.depth] * i)
            final_sum += 1-np.random.binomial(p=prob, n=1)
    return final_sum

cdef float __get_shpylm_1_minus_z_uwkj_sum(SHPYLMNode node):
    """
    Sample sum_{all segments}(sum_{all tables}(sum_{j=1..c_whk}(1-bernoulli((j-1)/(j-d)))) for the specific node
    where c_whk is count of customer at the specific table.
    
    Parameters
    ----------
    node : SHPYLMNode
        SHPYLM node of tree we want to add to our precomputed values
    Returns
    -------
    final_sum : float
        sum_{all segments}(sum_{all tables}(sum_{j=1..c_whk}(1-bernoulli((j-1)/(j-d))))
    """
    cdef float final_sum = 0.0
    cdef str segment
    cdef int c_whk
    cdef float prob
    cdef int j
    cdef NPYLM npylm = node.npylm
    cdef STables restaurant
    for segment in node.tables:
        restaurant = node.tables[segment]
        for c_whk in restaurant.tables:
            if c_whk >= 2:
                for j in range(1,c_whk):
                    prob = (j - 1) / (j - npylm.shpylm_ds[node.depth])
                    final_sum += 1 - np.random.binomial(p=prob, n=1)
    return final_sum

cdef float __get_shpylm_y_ui_sum(SHPYLMNode node):
    """
    Sample sum_{i=1..t_h}(bernoulli((theta)/(theta+d*i))) over all tables in the specific node.
    
    Parameters
    ----------
    node : SHPYLMNode
        SHPYLM node of tree we want to add to our precomputed values
    Returns
    -------
    final_sum : float
        sum_{i=1..t_h}(bernoulli((theta)/(theta+d*i)))
    """
    cdef float final_sum = 0.0
    cdef float prob
    cdef int i
    cdef NPYLM npylm = node.npylm
    if node.t_h >= 2:
        for i in range(1,node.t_h):
            prob = npylm.shpylm_thetas[node.depth] / \
                    (npylm.shpylm_thetas[node.depth] + npylm.shpylm_ds[node.depth] * i)
            final_sum += np.random.binomial(p=prob, n=1)
    return final_sum

cdef float __get_shpylm_logx_u(SHPYLMNode node):
    """
    Sample log(Beta(theta+1,c_h-1)) for the specific node.
    
    Parameters
    ----------
    node : SHPYLMNode
        SHPYLM node of tree we want to add to our precomputed values
    Returns
    -------
    logx_u : float
        log(Beta(theta+1,c_h-1)) for c_h >= 2, otherwise 0.0
    """
    cdef NPYLM npylm = node.npylm
    if node.c_h >= 2:
        return log(np.random.beta(npylm.shpylm_thetas[node.depth] + 1, node.c_h - 1))
    else:
        return 0.0



# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX           Tone Hierarchical Pitman-Yor Tree Hyperparameters           XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX




cdef void update_thpylm_d_theta(NPYLM npylm):
    """
    Update d and theta arrays of THPYLM in NPYLM.
    First recursivaly precompute sum(1-y_ui), sum(1-z_uwkj), sum(y_ui), sum(log(x_u))
    over all nodes for each depth separately.
    After that compute new 'd' using Beta priors as Beta(d_a+sum(1-y_ui), d_b+sum(1-z_uwkj))
        where d_a and d_b are defautly set hyperparameters
    And compute new 'theta' using Gamma priors as Gamma(theta_alpha+sum(u_i), 1/(theta_beta+sum(log(x_u))))
        where theta_alpha and theta_beta are defautly set hyperparameters
    for all shpylm tree depths.

    Parameters
    ----------
    npylm : NPYLM
        npylm wich we want to update and learn hyperparameters 
    """
    cdef int i
    cdef float alpha, beta
    # Init value arrays for d,theta updates
    cdef np.ndarray sum1_minus_y_ui = np.zeros([npylm.thpylm_max_depth+1], dtype=DTYPE)
    cdef np.ndarray sum1_minus_z_uwkj = np.zeros([npylm.thpylm_max_depth+1], dtype=DTYPE)
    cdef np.ndarray sumy_ui = np.zeros([npylm.thpylm_max_depth+1], dtype=DTYPE)
    cdef np.ndarray sumlogx_u = np.zeros([npylm.thpylm_max_depth+1], dtype=DTYPE)
    # Set the first depth
    sum1_minus_y_ui[0] = __get_thpylm_1_minus_y_ui(npylm.thpylm_root)
    sum1_minus_z_uwkj[0] = __get_thpylm_1_minus_z_uwkj_sum(npylm.thpylm_root)
    sumy_ui[0] = __get_thpylm_y_ui_sum(npylm.thpylm_root)
    sumlogx_u[0] = __get_thpylm_logx_u(npylm.thpylm_root)
    # Fill array values for all depths recursivaly
    __recursive_thpylm_d_theta_preparation(npylm.thpylm_root, sum1_minus_y_ui, sum1_minus_z_uwkj, sumy_ui, sumlogx_u)
    
    # Set new d, theta for all depths
    for i in range(npylm.thpylm_max_depth+1):
        # sample d
        alpha = npylm.d_a + sum1_minus_y_ui[i]
        beta = npylm.d_b + sum1_minus_z_uwkj[i]
        npylm.thpylm_ds[i] = np.random.beta(alpha, beta)
        # sample theta
        shape = npylm.theta_alpha + sumy_ui[i]
        scale = 1.0/(npylm.theta_beta - sumlogx_u[i])
        npylm.thpylm_thetas[i] = np.random.gamma(shape, scale)


cdef void __recursive_thpylm_d_theta_preparation(THPYLMNode node, np.ndarray sum1_minus_y_ui, 
                                                np.ndarray sum1_minus_z_uwkj, np.ndarray sumy_ui,
                                                np.ndarray sumlogx_u):
    """
    Recursivaly precompute sum(1-y_ui), sum(1-z_uwkj), sum(y_ui), sum(log(x_u))
    over all nodes for each depth separately.

    Parameters
    ----------
    node : THPYLMNode
        THPYLM node of tree we want to add to our precomputed values
    sum1_minus_y_ui : np.ndarray
        array of floats of sum(1-y_ui) for all depths
    sum1_minus_z_uwkj : np.ndarray
        array of floats of sum(1-z_uwkj) for all depths
    sumy_ui : np.ndarray
        array of floats of sum(y_ui) for all depths
    sumlogx_u : np.ndarray
        array of floats of sum(log(x_u)) for all depths
    """
    cdef THPYLMNode child
    cdef str context
    for context in node.children:
        child = node.children[context]
        # Update arrays
        sum1_minus_y_ui[child.depth] += __get_thpylm_1_minus_y_ui(child)
        sum1_minus_z_uwkj[child.depth] += __get_thpylm_1_minus_z_uwkj_sum(child)
        sumy_ui[child.depth] += __get_thpylm_y_ui_sum(child)
        sumlogx_u[child.depth] += __get_thpylm_logx_u(child)
        # Recursive call
        __recursive_thpylm_d_theta_preparation(child, sum1_minus_y_ui, sum1_minus_z_uwkj, sumy_ui, sumlogx_u)


cdef float __get_thpylm_1_minus_y_ui(THPYLMNode node):
    """
    Sample sum_{i=1..t_h}(1-bernoulli((theta)/(theta+d*i))) over all tables in the specific node.
    
    Parameters
    ----------
    node : THPYLMNode
        THPYLM node of tree we want to add to our precomputed values
    Returns
    -------
    final_sum : float
        sum_{i=1..t_h}(1-bernoulli((theta)/(theta+d*i)))
    """
    cdef float final_sum = 0.0
    cdef float prob
    cdef int i
    cdef NPYLM npylm = node.npylm
    if node.t_h >= 2:
        for i in range(1,node.t_h):
            prob = npylm.thpylm_thetas[node.depth] / \
                    (npylm.thpylm_thetas[node.depth] + npylm.thpylm_ds[node.depth] * i)
            final_sum += 1-np.random.binomial(p=prob, n=1)
    return final_sum

cdef float __get_thpylm_1_minus_z_uwkj_sum(THPYLMNode node):
    """
    Sample sum_{all segments}(sum_{all tables}(sum_{j=1..c_whk}(1-bernoulli((j-1)/(j-d)))) for the specific node
    where c_whk is count of customer at the specific table.
    
    Parameters
    ----------
    node : THPYLMNode
        THPYLM node of tree we want to add to our precomputed values
    Returns
    -------
    final_sum : float
        sum_{all segments}(sum_{all tables}(sum_{j=1..c_whk}(1-bernoulli((j-1)/(j-d))))
    """
    cdef float final_sum = 0.0
    cdef str segment
    cdef int c_whk
    cdef float prob
    cdef int j
    cdef NPYLM npylm = node.npylm
    cdef TTables restaurant
    for segment in node.tables:
        restaurant = node.tables[segment]
        for c_whk in restaurant.tables:
            if c_whk >= 2:
                for j in range(1,c_whk):
                    prob = (j - 1) / (j - npylm.thpylm_ds[node.depth])
                    final_sum += 1 - np.random.binomial(p=prob, n=1)
    return final_sum

cdef float __get_thpylm_y_ui_sum(THPYLMNode node):
    """
    Sample sum_{i=1..t_h}(bernoulli((theta)/(theta+d*i))) over all tables in the specific node.
    
    Parameters
    ----------
    node : THPYLMNode
        THPYLM node of tree we want to add to our precomputed values
    Returns
    -------
    final_sum : float
        sum_{i=1..t_h}(bernoulli((theta)/(theta+d*i)))
    """
    cdef float final_sum = 0.0
    cdef float prob
    cdef int i
    cdef NPYLM npylm = node.npylm
    if node.t_h >= 2:
        for i in range(1,node.t_h):
            prob = npylm.thpylm_thetas[node.depth] / \
                    (npylm.thpylm_thetas[node.depth] + npylm.thpylm_ds[node.depth] * i)
            final_sum += np.random.binomial(p=prob, n=1)
    return final_sum

cdef float __get_thpylm_logx_u(THPYLMNode node):
    """
    Sample log(Beta(theta+1,c_h-1)) for the specific node.
    
    Parameters
    ----------
    node : THPYLMNode
        THPYLM node of tree we want to add to our precomputed values
    Returns
    -------
    logx_u : float
        log(Beta(theta+1,c_h-1)) for c_h >= 2, otherwise 0.0
    """
    cdef NPYLM npylm = node.npylm
    if node.c_h >= 2:
        return log(np.random.beta(npylm.thpylm_thetas[node.depth] + 1, node.c_h - 1))
    else:
        return 0.0