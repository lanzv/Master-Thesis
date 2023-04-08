from npylm cimport NPYLM
from chant cimport EOC, BOC
import numpy as np
cimport numpy as np
np.import_array()
DTYPE = np.float64


cdef list viterbi_segment_chants(NPYLM npylm, list chants_str):
    """
    Process viterbi segmentation for all testing chants based on the hpylm model.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    chant : list of strings
        list of current training chants we are sampling, their segmentation will be changed
    Returns
    -------
    final_segmetation : list of lists of strings
        list of segmented chants, one chant segmentation is represented as list of strigns(~segments)
    """
    cdef list final_segmentation = []
    for chant_string in chants_str:
        final_segmentation.append(__get_chant_segmentation(npylm, chant_string))
    return final_segmentation



cdef list __get_chant_segmentation(NPYLM npylm, str chant_str):
    """
    Process the forward filtering to get precomputed alpha array.
    With the alpha values, do the backward sampling to predict the segmentation.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    chant_str : string
        current chant we are sampling, its segmentation will be changed after this function
    Returns
    -------
    final_segmetation : list of lists of strings
        list of segmented chants, one chant segmentation is represented as list of strigns(~segments)
    """

    cdef np.ndarray alpha = __forward_filtering(npylm, chant_str)
    return __get_segments_with_backward_sampling(npylm, chant_str, alpha)


cdef np.ndarray __forward_filtering(NPYLM npylm, str chant_str):
    """
    Process the forward filtering and precompute alpha array with marginalized probabilities
    for all t, k (t is the position in chant, k is the length of the last segment).
    The scaling is used for avoiding of underflowing. The original paper used expsumlog().

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    chant_str : string
        current chant we are sampling, its segmentation will be changed after this function
    Returns
    -------
    alpha : np.ndarray
        precompute alpha[t,k] array with marginalized probabilities for all t, k 
        (t is the position in chant, k is the length of the last segment)
    """
    cdef int chant_len = len(chant_str)
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
                    npylm.get_bigram_probability(BOC, chant_str[t-k:t])\
                    * alpha[0, 0]\
                    * prod_scaling
                    )
            else:
                # for j in range(1, t-k) in the original word segmentation paper - we have to consider max_size
                for j in range(1, min(max_segment_size, t-k)+1): 
                    sum_prob += (
                        # first gram: (t-k-j+1):(t-k), second gram: (t-k+1):(t) (by the "vector indexing")
                        npylm.get_bigram_probability(chant_str[t-k-j:t-k], chant_str[t-k:t])\
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


cdef list __get_segments_with_backward_sampling(NPYLM npylm, str chant_str, np.ndarray alpha):
    """
    Get segmentation using the backward sampling regarding the precompued alpha.
    The final segmentation is chosen
    based on argmax of k ~ p(w_{i} | c_{t-k+1}^{t}, Theta) * alpha[t][k] in each step.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    chant_str : string
        current chant we are sampling, its segmentation will be changed after this function
    alpha : np.ndarray
        precompute alpha[t,k] array with marginalized probabilities for all t, k 
        (t is the position in chant, k is the length of the last segment)
    Returns
    -------
    final_segmetation : list of lists of strings
        list of segmented chants, one chant segmentation is represented as list of strigns(~segments)
    """
    cdef int t = len(chant_str)
    cdef int max_segment_size = npylm.max_segment_size
    cdef list segmented_chant = []
    cdef int i, j, k
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
                npylm.get_bigram_probability(chant_str[t-k:t], w) * alpha[t, k]
            )
        k = k_candidates[np.argmax(probs)]
        w = chant_str[t-k:t]
        t -= k
        borders.append(t)


    borders.reverse()
    for i, j in zip(borders[:-1], borders[1:]):
        segmented_chant.append(chant_str[i:j])

    return segmented_chant