cdef str BOC = "<boc>" # beginning of chant
cdef str EOC = "$" # end of chant, because of the code logic, it needs to be only one character long
cdef str BOS = "<bos>" # beginning of segment
cdef str EOS = "<eos>" # end of segment

cdef class Chant():
    def __cinit__(self, str chant_string):
        """
        Constructor of Chant data structure.

        Parameters
        ----------
        chant_string : string
            chant string in its very basic form (no prefixes/suffixes/..)
        """
        self.chant_string = chant_string
        self.segmentation = [chant_string]


    cdef void set_segmentation(self, list borders):
        """
        Function that get list of borders of segmetns and it will reupload chant's segmentation
        regarding those borders.

        example:
            borders=[0, 3, 4, 10], chant_string="abcdefghij" -> segmentation=["abc", "d", "efghij"]

        Parameters
        ----------
        borders : list of ints
            list of indices of chant_string string chars (in other word char array),
            that describe segment borders - they have to be order by ascending
        """
        cdef list segmented_chant = []
        cdef int i, j
        for i, j in zip(borders[:-1], borders[1:]):
            segmented_chant.append(self.chant_string[i:j])
        self.segmentation = segmented_chant