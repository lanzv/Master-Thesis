"""
Model that generate overlapping ngrams of all chants.
"""
class OverlappigNGrams:

    def __init__(self, ngram=4):
        self.ngram = ngram

    def predict_segments(self, chants):
        """
        Generate overlapping ngrams of each chant string melody.
        The N of N-gram is specified in constructor.

        Parameters
        ----------
        chants : list of strings
            list of chants, each chant is represented as string melody
        Returns
        -------
        overlapping_ngrams : list of lists of strings
            list of overlappi
        """
        overlapping_ngrams = []
        for chant_string in chants:
            chant_segments = []
            if len(chant_string) < self.ngram:
                chant_segments.append(chant_string)
            for i in range(0, len(chant_string)-self.ngram+1):
                segment = chant_string[i:i+self.ngram]
                chant_segments.append(segment)
            overlapping_ngrams.append(chant_segments)
        return overlapping_ngrams