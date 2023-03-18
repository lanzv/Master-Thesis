import numpy as np
from collections import Counter

class FinalRangeClassifier():
    @staticmethod
    def predict(chants):
        predicted_modes = []
        for chant_string in chants:
            assert type(chant_string) is str or type(chant_string) is np.str_
            predicted_modes.append(FinalRangeClassifier.__process_chant(chant_string[-1], chant_string))
        return predicted_modes

    @staticmethod
    def __process_chant(finalis, chant_string):
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