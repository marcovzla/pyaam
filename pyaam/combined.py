# coding: utf-8

from __future__ import division

import numpy as np
from pyaam.shape import ShapeModel
from pyaam.texture import TextureModel
from pyaam.texturemapper import TextureMapper
from pyaam.utils import pca

class CombinedModel(object):
    def __init__(self, model, variance, weights):
        self.model = model
        self.variance = variance
        self.weights = weights

    @classmethod
    def train(cls, lmks, imgs, ref, frac, kmax):
        smodel = ShapeModel.load('data/shape.npz')
        tmodel = TextureModel.load('data/texture.npz')

        # build diagonal matrix of weights that measures
        # the unit difference between shape and texture parameters
        r = tmodel.variance.sum() / smodel.variance.sum()
        Ws = r * np.eye(smodel.num_modes())

        n_samples = lmks.shape[1]

        tm = TextureMapper(480, 640)

        # create empty matrix
        n_feats = smodel.num_modes() + tmodel.num_modes()
        A = np.empty((n_feats, n_samples))

        # concatenate shape and texture parameters for each training example
        for i in range(n_samples):
            print("Sample ",i," of ",n_samples," samples...")
            img = next(imgs)
            lmk = lmks[:,i]
            sparams = smodel.calc_params(lmk)
            tparams = tmodel.calc_params(img, lmk, ref, tm.warp_triangles)
            # ignore first 4 shape parameters
            A[:,i] = np.concatenate((Ws.dot(sparams[4:]), tparams))

        D = pca(A, frac, kmax)

        # compute variance
        Q = pow(D.T.dot(A), 2)
        e = Q.sum(axis=1) / (n_samples-1)

        return cls(D, e, Ws)

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        return cls(data['model'], data['variance'], data['weights'])

    def save(self, filename):
        np.savez(filename, model=self.model, variance=self.variance, weights=self.weights)

    def num_modes(self):
        return self.model.shape[1]

    def calc_shp_tex_params(self, params, split):
        st_params = self.model.dot(params)
        sparams = np.linalg.inv(self.weights).dot(st_params[:split])
        tparams = st_params[split:]
        return sparams, tparams
