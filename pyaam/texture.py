# coding: utf-8

from __future__ import division

import cv2
import numpy as np
from pyaam.texturemapper import TextureMapper
from pyaam.utils import get_mask, get_aabb, get_vertices, normalize, pca



class TextureModel(object):
    def __init__(self, model, mean, variance):
        self.model = model
        self.mean = mean
        self.variance = variance

    @classmethod
    def train(cls, lmks, imgs, ref, frac, kmax):
        G = get_data_matrix(imgs, lmks, ref)
        Gm = G.mean(axis=1)
        G -= Gm[:,np.newaxis]
        N = lmks.shape[1]
        D = pca(G, frac, kmax)
        # normalize eigenvectors
        for i in xrange(D.shape[1]):
            D[:,i] /= np.linalg.norm(D[:,i])
        # compute variance
        Q = D.T.dot(G)
        Q = pow(Q, 2)
        e = Q.sum(axis=1) / (N-1)
        return cls(D, Gm, e)

    @classmethod
    def load(cls, filename):
        arch = np.load(filename)
        return cls(arch['model'], arch['mean'], arch['variance'])

    def save(self, filename):
        np.savez(filename, model=self.model, mean=self.mean, variance=self.variance)

    def num_modes(self):
        return self.model.shape[1]

    def calc_texture(self, params):
        t = self.mean + self.model.dot(params)
        return t.clip(0, 255)  # clamp pixels intensities

    def calc_params(self, img, lmk, ref, tm):
        ref = ref.astype('int32')
        src = lmk.reshape(ref.shape)
        img = normalize(img, get_aabb(src))
        mask = get_mask(ref, img.shape[:2])
        verts = get_vertices(ref)
        warp = tm.warp_triangles(img, src[verts], ref[verts])
        t = warp[mask].ravel() - self.mean
        p = self.model.T.dot(t)
        # clamp
        c = 3
        for i in xrange(len(self.variance)):
            v = c * np.sqrt(self.variance[i])
            if abs(p[i]) > v:
                p[i] = v if p[i] > 0 else -v
        return p



def get_data_matrix(imgs, lmks, ref):
    ref = ref.reshape((ref.size//2, 2)).astype('int32')
    mask = get_mask(ref, (640, 480))  # FIXME hardcoded image size
    verts = get_vertices(ref)
    tm = TextureMapper(480, 640)  # ditto
    n_samples = lmks.shape[1]
    n_pixels = mask.sum() * 3
    G = np.empty((n_pixels, n_samples))
    for i in xrange(n_samples):
        src = lmks[:,i].reshape(ref.shape)
        img = normalize(next(imgs), get_aabb(src))
        warp = tm.warp_triangles(img, src[verts], ref[verts])
        G[:,i] = warp[mask].ravel()
    return G
