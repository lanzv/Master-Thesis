from npylm cimport NPYLM
from chant cimport Chant, EOC, BOC
from random_utils cimport random_choice
import numpy as np
cimport numpy as np
np.import_array()
DTYPE = np.float64


cdef void blocked_gibbs_iteration(NPYLM npylm, list chants):
    """
    Process one blocked gibbs iteration for all training chants.
    First shuffle chants randomly. Sample each chant - remove chant segments
    from NPYLM, get new segmentation, add new chant segments to NPYLM.
    At the end of all chant sampling also sample hyperparameters.

    Parameters
    ----------
    hpylm : NPYLM
        current state of hpylm language model
    chant : list
        list of current training chants we are sampling, their segmentation will be changed
    """
    cdef int chant_id
    cdef np.ndarray rand_indices = np.arange(len(chants))

    np.random.shuffle(rand_indices)
    for chant_id in rand_indices:
        npylm.remove_chant(chants[chant_id])
        __optmize_chant_segmentation(npylm, chants[chant_id])
        npylm.add_chant(chants[chant_id])

    __sample_hyperparameters(npylm)


cdef void __optmize_chant_segmentation(NPYLM npylm, Chant chant):
    """
    Process the forward filtering to get precomputed alpha array.
    With the alpha values, do the backward sampling to create a new segmentation.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    chant : Chant
        current chant we are sampling, its segmentation will be changed after this function
    """

    cdef np.ndarray alpha = __forward_filtering(npylm, chant)
    __backward_sampling(npylm, chant, alpha)


cdef np.ndarray __forward_filtering(NPYLM npylm, Chant chant):
    """
    Process the forward filtering and precompute alpha array with marginalized probabilities
    for all t, k (t is the position in chant, k is the length of the last segment).
    The scaling is used for avoiding of underflowing. The original paper used expsumlog().

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    chant : Chant
        current chant we are sampling, its segmentation will be changed after this function
    Returns
    -------
    alpha : np.ndarray
        precompute alpha[t,k] array with marginalized probabilities for all t, k 
        (t is the position in chant, k is the length of the last segment)
    """
    cdef int chant_len = len(chant.chant_string)
    cdef int max_segment_size = npylm.max_segment_size
    cdef np.ndarray alpha = np.zeros([chant_len+1, max_segment_size+1], dtype=DTYPE)
    cdef np.ndarray scaling_alpha = np.zeros([chant_len+1], dtype=DTYPE)
    cdef int t, k, j
    cdef float sum_prob
    cdef float prod_scaling
    cdef float sum_alpha_t

    alpha[0, 0] = 1.0
    for t in range(1, chant_len+1):
        prod_scaling = 1.0
        sum_alpha_t = 0.0
        for k in range(1, min(max_segment_size, t)+1):
            if k != 1:
                prod_scaling *= scaling_alpha[t-k+1]
            sum_prob = 0.0
            if t-k == 0:
                # first gram is an <boc> beggining of chant
                sum_prob += (\
                    # second gram: (t-k+1):(t) (by the "vector indexing")
                    npylm.get_bigram_probability(BOC, chant.chant_string[t-k:t])\
                    * alpha[0, 0]\
                    * prod_scaling
                    )
            else:
                # for j in range(1, t-k) in the original word segmentation paper - we have to consider max_size
                for j in range(1, min(max_segment_size, t-k)+1): 
                    sum_prob += (
                        # first gram: (t-k-j+1):(t-k), second gram: (t-k+1):(t) (by the "vector indexing")
                        npylm.get_bigram_probability(chant.chant_string[t-k-j:t-k], chant.chant_string[t-k:t])\
                        * alpha[t-k, j]\
                        * prod_scaling
                        )
            alpha[t, k] = sum_prob
            sum_alpha_t += sum_prob

        # Perform scaling to avoid underflowing
        for k in range(1, min(max_segment_size, t)+1):
            alpha[t, k] /= sum_alpha_t
        scaling_alpha[t] = 1.0/sum_alpha_t

    return alpha


cdef void __backward_sampling(NPYLM npylm, Chant chant, np.ndarray alpha):
    """
    Do the backward sampling to get optimized segmentation with np.random.choice of 
    all k candidates k ~ p(w_{i} | c_{t-k+1}^{t}, Theta) * alpha[t][k] in each step.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    chant : Chant
        current chant we are sampling, its segmentation will be changed after this function
    alpha : np.ndarray
        precompute alpha[t,k] array with marginalized probabilities for all t, k 
        (t is the position in chant, k is the length of the last segment).
    """
    cdef int t = len(chant.chant_string)
    cdef int max_segment_size = npylm.max_segment_size
    cdef int k
    cdef list probs
    cdef list k_candidates
    cdef list borders = [t]
    cdef str w = EOC

    while t > 0:   
        probs = []
        k_candidates = []
        for k in range(1, min(max_segment_size, t)+1):
            k_candidates.append(k)
            probs.append(
                # first gram: (t-k+1):(k), second gram: w (by the "vector indexing")
                npylm.get_bigram_probability(chant.chant_string[t-k:t], w) * alpha[t, k]
            )
        k = k_candidates[random_choice(probs)]
        w = chant.chant_string[t-k:t]
        t -= k
        borders.append(t)

    
    borders.reverse()
    chant.set_segmentation(borders)


cdef void __sample_hyperparameters(NPYLM npylm):
    """
    ToDo

    Parameters
    ----------
    npylm : 

    chant : 

    alpha : 
    """
    pass