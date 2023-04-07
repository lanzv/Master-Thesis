cdef str BOC
cdef str EOC
cdef str BOS
cdef str EOS
cdef class Chant():
    cdef str chant_string
    cdef list segmentation
    cdef void set_segmentation(self, list borders)