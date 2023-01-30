



class OverlappigNGrams:

    def __init__(self, ngram=4):
        self.ngram = ngram

    def predict_segments(self, chants):
        segmentations = []
        for chant_string in chants:
            chant_segments = []
            if len(chant_string) < self.ngram:
                chant_segments.append(chant_string)
            for i in range(0, len(chant_string)-self.ngram+1):
                segment = chant_string[i:i+self.ngram]
                chant_segments.append(segment)
            segmentations.append(chant_segments)
        return segmentations