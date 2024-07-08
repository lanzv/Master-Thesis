import numpy as np
import random
from src.eval.maww_score import mawp_score

class RandomModel():
    def __init__(self, min_size, max_size):
        """
        Constructor of RandomModel - the model that predicts segments randomly.

        Parameters
        ----------
        min_size : int
            minimal size of segments
        max_size : int
            maximal size of segments 
        """
        self.min_size = min_size
        self.max_size = max_size
    
    def predict_segments(self, chants, mu=5, sigma=2):
        """
        Predict random segmentation based on gaussian distribution.
        All predicted segments are clipped to be in the range of model's min_size and max_size.

        Parameters
        ----------
        chants : list of strings
            chants represented as string (no spaces, no segments)
        mu : float
            mu value for gaussian random distribution
        sigma : float
            sigma value for gaussian random distribution
        Returns
        -------
        rand_segments : list of list of strings
            list of chants represnted as list of string segments
        perplexity : float
            we ignore the value in this model since we cannot compute the perplexity of random model
        """
        rand_segments = []
        for chant in chants:
            new_chant_segments = []
            i = 0
            while i != len(chant):
                # Find new segment
                new_len = np.clip(a = int(random.gauss(mu, sigma)),
                    a_min = self.min_size, a_max = self.max_size)
                k = min(i+new_len, len(chant))
                new_chant_segments.append(chant[i:k])
                # Update i index
                i = k
            rand_segments.append(new_chant_segments)
        return rand_segments, -1

    def get_mawp_score(self):
        """
        Compute Melody Aligned With Phrases score of this model.

        Returns
        -------
        mawp_score : float
            mawp score of this model
        """
        return mawp_score(self)