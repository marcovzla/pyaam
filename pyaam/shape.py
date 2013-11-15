# coding: utf-8

from __future__ import division

import numpy as np
from pyaam.utils import pca



class ShapeModel(object):
    def __init__(self, model, variance):
        self.model = model
        self.variance = variance

    @classmethod
    def train(cls, X, frac, kmax):
        n_samples = X.shape[1]
        n_points = X.shape[0] // 2
        # align shapes
        Y = procrustes(X)
        # compute rigid transform
        R = calc_rigid_basis(Y)
        # project out rigidity
        P = R.T.dot(Y)
        dY = Y - R.dot(P)
        # compute non-rigid transformation
        D = pca(dY, frac, min(kmax, n_samples-1, n_points-1))
        k = D.shape[1]
        # combine subspaces
        V = np.concatenate((R,D), axis=1)
        # project raw data onto subspace
        Q = V.T.dot(X)
        # normalize coordinates w.r.t. scale
        for i in xrange(n_samples):
            Q[:,i] /= Q[0,i]
        # compute variance
        e = np.empty(4+k, dtype=float)
        Q = pow(Q, 2)
        e[:4] = -1  # no clamping for rigid body coefficients
        e[4:] = Q[4:].sum(axis=1) / (n_samples-1)
        # return model
        return cls(V, e)

    @classmethod
    def load(cls, fname):
        arch = np.load(fname)
        return cls(arch['model'], arch['variance'])

    def save(self, fname):
        np.savez(fname, model=self.model, variance=self.variance)

    def num_modes(self):
        return self.model.shape[1] - 4

    def calc_shape(self, params):
        return self.model.dot(params)

    def get_shape(self, scale=1, tranx=0, trany=0):
        params = self.get_params(scale, tranx, trany)
        return self.calc_shape(params)

    def calc_params(self, pts, c_factor=3.0):
        params = self.model.T.dot(pts)
        return self.clamp(params, c_factor)

    def get_params(self, scale=1, tranx=0, trany=0):
        # compute rigid parameters
        n = self.model.shape[0] // 2
        scale /= self.model[::2,0].ptp()
        tranx *= n / self.model[:,2].sum()
        trany *= n / self.model[:,3].sum()
        p = np.zeros(self.model.shape[1])
        p[0] = scale
        p[2] = tranx
        p[3] = trany
        return p

    def clamp(self, p, c):
        scale = p[0]
        for i in xrange(len(p)):
            var = self.variance[i]
            if var < 0:
                # ignore rigid components
                continue
            v = c * np.sqrt(var)
            # preserve sign of coordinate
            if abs(p[i] / scale) > v:
                p[i] = v * scale if p[i] > 0 else -v * scale
        return p


def procrustes(X, max_iters=100, tolerance=1e-6):
    """removes global rigid motion from a collection of shapes"""
    n_samples = X.shape[1]
    n_points = X.shape[0] // 2
    # copy of data to work on
    P = X.copy()

    # remove center of mass of each shape's instance
    P[::2,:] -= P[::2,:].sum(axis=0) / n_points
    P[1::2,:] -= P[1::2,:].sum(axis=0) / n_points

    # optimize scale and rotation
    C_old = None
    for _ in xrange(max_iters):
        # compute normalized canonical shape
        C = P.sum(axis=1) / n_samples
        C /= np.linalg.norm(C)

        # are we done?
        if C_old is not None and np.linalg.norm(C - C_old) < tolerance:
            break

        # keep copy of current estimate of canonical shape
        C_old = C.copy()

        # rotate and scale each shape to best match canonical shape
        for i in xrange(n_samples):
            R = rot_scale_align(P[:,i], C)
            pts = np.row_stack((P[::2,i], P[1::2,i]))
            P[:,i] = R.dot(pts).T.flatten()

    # return procrustes aligned shapes
    return P

def rot_scale_align(src, dst):
    """computes the in-place rotation and scaling that best aligns
    shape instance `src` to shape instance `dst`"""
    # separate x and y
    srcx, srcy = src[::2], src[1::2]
    dstx, dsty = dst[::2], dst[1::2]
    # construct and solve linear system
    d = sum(pow(src, 2))
    a = sum(srcx*dstx + srcy*dsty) / d
    b = sum(srcx*dsty - srcy*dstx) / d
    # return scale and rotation matrix
    return np.array([[a,-b],[b,a]])

def calc_rigid_basis(X):
    """model global transformation as linear subspace"""
    n_samples = X.shape[1]
    n_points = X.shape[0] // 2

    # compute canonical shape
    mean = X.mean(axis=1)

    # construct basis for similarity transform
    R = np.empty((2*n_points, 4), dtype=float)
    R[::2,0] = mean[::2]
    R[1::2,0] = mean[1::2]
    R[::2,1] = -mean[1::2]
    R[1::2,1] = mean[::2]
    R[::2,2] = 1
    R[1::2,2] = 0
    R[::2,3] = 0
    R[1::2,3] = 1

    return gram_schmid(R)

def gram_schmid(V):
    """Gram-Schmid orthonormalization (in-place)"""
    n = V.shape[1]
    for i in xrange(n):
        for j in xrange(i):
            # subtract projection
            V[:,i] -= np.dot(V[:,i], V[:,j]) * V[:,j]
        # normalize
        V[:,i] /= np.linalg.norm(V[:,i])
    # orthonormalization was performed in-place
    # but we return V for convenience
    return V
