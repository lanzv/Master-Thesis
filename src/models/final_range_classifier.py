import numpy as np
from collections import Counter

"""
Final pitch + melody range classifier of mode.
Really simple approach to predict modes based on chant melodies.
The accuracy of the model is not that great (something around 45%).
"""
class FinalRangeClassifier():
    @staticmethod
    def predict(chants):
        """
        Predict modes of comming chants using the final tone and melody range.
        Unfortinutelly the final tone of the string melody is not always the "finalis" tone,
        because of the final extra melodies and variances at the end of chants.

        Parameters
        ----------
        chants : list of strings
            list of chants, each chant is represented as string melody
        Returns
        -------
        predicted_modes : list of chars
            list of predicted modes
        """
        predicted_modes = []
        for chant_string in chants:
            assert type(chant_string) is str or type(chant_string) is np.str_
            predicted_modes.append(FinalRangeClassifier.__process_chant(chant_string[-1], chant_string))
        return predicted_modes

    @staticmethod
    def __process_chant(finalis, chant_string):
        """
        Predict mode based on the final tone and chant string melody.
        Based on the final tone, one of four pairs 1,2 3,4 5,6 7,8 is chosen.
        The specific mode of the pair is then chosen based on the range regarding the gregorian chant theory.

        Parameters
        ----------
        finalis : char
            last tone of the chant melody
        chant_string : string
            the chant melody
        Returns
        -------
        mode : char
            mode prediction
        """
        counter = Counter(chant_string)
        if finalis == 'd':
            dorian_sum = counter['d'] + counter['e'] + counter['f'] + counter['g'] + counter['h'] + counter['j']+ counter['k']+ counter['l']
            hyperdorian_sum = counter['a'] + counter['b'] + counter['c'] + counter['d'] + counter['e'] + counter['f']+ counter['g']+ counter['h']
            if hyperdorian_sum < dorian_sum:
                return '1'
            else:
                return '2'
        elif finalis == 'e':
            phrygian_sum = counter['e'] + counter['f'] + counter['g'] + counter['h'] + counter['j']+ counter['k']+ counter['l'] + counter['m']
            hyperphrygian_sum = counter['b'] + counter['c'] + counter['d'] + counter['e'] + counter['f']+ counter['g']+ counter['h'] + counter['j']
            if hyperphrygian_sum < phrygian_sum:
                return '3'
            else:
                return '4'
        elif finalis == 'f':
            lydian_sum = counter['f'] + counter['g'] + counter['h'] + counter['j']+ counter['k']+ counter['l'] + counter['m'] + counter['n'] 
            hyperlydian_sum = counter['c'] + counter['d'] + counter['e'] + counter['f']+ counter['g']+ counter['h'] + counter['j'] + counter['k']
            if hyperlydian_sum < lydian_sum:
                return '5'
            else:
                return '6'
        elif finalis == 'g':
            mixolydian_sum = counter['g'] + counter['h'] + counter['j']+ counter['k']+ counter['l'] + counter['m'] + counter['n'] + counter['o']  
            hypermixolydian_sum = counter['d'] + counter['e'] + counter['f']+ counter['g']+ counter['h'] + counter['j'] + counter['k'] + counter['l']
            if hypermixolydian_sum < mixolydian_sum:
                return '7'
            else:
                return '8'
        else:
            return '1'