# cython: profile=False
# cython: language_level=3
# cython: linetrace=False
cimport cython
from libc.stdlib cimport rand, RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int fast_sample(x):
    assert x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1)
    x = x.detach().cpu().contiguous()
    x = x.numpy() if x.ndim == 1 else x[0].numpy()
    cdef float[:] x_view = x
    cdef float* x_ = &x_view[0]
    cdef float total = 0.
    cdef float r
    cdef float acc = 0
    cdef int i
    cdef int n = x.shape[0]

    for i in range(n):
        total += x_[i]

    r = rand() / RAND_MAX * total
    for i in range(n):
        acc += x_[i]
        if acc >= r:
            return i
    print(x)
    assert False, ("Should not reach here. Called fast_sample on "
        "an invalid distribution (all zeros or negative values)")

