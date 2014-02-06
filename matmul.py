########################################################################
#
#       License: BSD
#       Created: October 11, 2013
#       Author:  Francesc Alted
#
########################################################################

"""
Implementation of an out of core matrix-matrix multiplication for PyTables.
"""

import sys, math

import numpy as np
import tables as tb

_MB = 2**20
OOC_BUFFER_SIZE = 32*_MB
"""The buffer size for out-of-core operations.
"""

def dot(a, b, out=None):
    """
    Matrix multiplication of two 2-D arrays.

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : array_like, optional
        Output argument. This must have the exact kind that would be
        returned if it was not used.

    Returns
    -------
    output : CArray or scalar
        Returns the dot product of `a` and `b`.  If `a` and `b` are
        both scalars or both 1-D arrays then a scalar is returned;
        otherwise a new CArray (in file dot.h5:/out) is returned.  If
        `out` parameter is provided, then it is returned instead.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as the
        second-to-last dimension of `b`.
    """

    if len(a.shape) != 2 or len(b.shape) != 2:
        raise (ValueError, "only 2-D matrices supported")

    if a.shape[1] != b.shape[0]:
        raise (ValueError,
               "last dimension of `a` does not match first dimension of `b`")

    l, m, n = a.shape[0], a.shape[1], b.shape[1]

    if out:
        if out.shape != (l, n):
            raise (ValueError, "`out` array does not have the correct shape")
    else:
        f = tb.openFile('dot.h5', 'w')
        filters = tb.Filters(complevel=5, complib='blosc')
        out = f.createCArray(f.root, 'out', tb.Atom.from_dtype(a.dtype),
                             shape=(l, n), filters=filters)

    # Compute a good block size
    buffersize = OOC_BUFFER_SIZE
    bl = math.sqrt(buffersize / out.dtype.itemsize)
    bl = 2**int(math.log(bl, 2))
    for i in range(0, l, bl):
        for j in range(0, n, bl):
            for k in range(0, m, bl):
                a0 = a[i:min(i+bl, l), k:min(k+bl, m)]
                b0 = b[k:min(k+bl, m), j:min(j+bl, n)]
                out[i:i+bl, j:j+bl] += np.dot(a0, b0)

    return out

if __name__ == "__main__":
    """Small benchmark for comparison against numpy.dot() speed"""
    from time import time

    # Matrix dimensions
    L, M, N = 1000, 100, 2000
    print "Multiplying (%d, %d) x (%d, %d) matrices" % (L, M, M, N)

    a = np.linspace(0, 1, L*M).reshape(L, M)
    b = np.linspace(0, 1, M*N).reshape(M, N)

    t0 = time()
    cdot = np.dot(a,b)
    print "Time for np.dot->", round(time()-t0, 3), cdot.shape

    f = tb.openFile('matrix-pt.h5', 'w')

    l, m, n = a.shape[0], a.shape[1], b.shape[1]

    filters = tb.Filters(complevel=5, complib='blosc')
    ad = f.createCArray(f.root, 'a', tb.Float64Atom(), (l,m),
                        filters=filters)
    ad[:] = a
    bd = f.createCArray(f.root, 'b', tb.Float64Atom(), (m,n),
                        filters=filters)
    bd[:] = b
    cd = f.createCArray(f.root, 'c', tb.Float64Atom(), (l,n),
                        filters=filters)

    t0 = time()
    dot(a, b, out=cd)
    print "Time for ooc dot->", round(time()-t0, 3), cd.shape

    np.testing.assert_almost_equal(cd, cdot)

    f.close()

