#!/usr/bin/env python

from __future__ import division

import numpy as np
import tables as tb
import matmul

print 'opening file ...'
h5 = tb.openFile('/media/disk2/perturbations.h5', mode='r')
print 'get G ...'
G = h5.root.residuals
print 'read P ...'
P = h5.root.perturbations[:]
print 'P pseudoinverse ...'
P_pinv = np.linalg.pinv(P)
print 'alloc J'
rows = G.shape[0]
cols = P_pinv.shape[1]
J = np.zeros((rows, cols))
print 'J = G * P_pinv'
matmul.dot(G, P_pinv, out=J)
print 'J pseudoinverse ...'
R = np.linalg.pinv(J)
print 'writing file ...'
np.savez('data/regmat.npz', R=R)
