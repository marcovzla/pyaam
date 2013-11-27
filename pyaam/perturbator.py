# coding: utf-8

from __future__ import division

import numpy as np

class Perturbator(object):
    def __init__(self, s_sigma, t_sigma):
        self.s_sigma = s_sigma
        self.t_sigma = t_sigma

        self.scale_perts = (0.95, 0.97, 0.99, 1.01, 1.03, 1.05)
        self.angle_perts = [np.radians(x) for x in (-5, -3, -1, 1, 3, 5)]
        self.trans_perts = (-6, -3, -1, 1, 3, 6)
        self.param_perts = (-0.5, -0.25, 0.25, 0.5)

    def num_perts(self):
        n_perts = 0
        n_perts += len(self.scale_perts)
        n_perts += len(self.angle_perts)
        n_perts += len(self.trans_perts) * 2
        n_params = len(self.s_sigma) + len(self.t_sigma)
        n_perts += len(self.param_perts) * n_params
        return n_perts

    def perturbations(self, shape, texture):
        pose = shape[:4]
        shape = shape[4:]

        a, b = pose[:2]
        scale = np.sqrt(a*a + b*b)
        angle = np.arctan2(b, a)

        for pert in self.scale_perts:
            pert_pose = pose.copy()
            pert_pose[0] = scale * pert * np.cos(angle)
            pert_pose[1] = scale * pert * np.sin(angle)
            yield np.concatenate((pert_pose, shape, texture))

        for pert in self.angle_perts:
            pert_pose = pose.copy()
            pert_pose[0] = scale * np.cos(angle + pert)
            pert_pose[1] = scale * np.sin(angle + pert)
            yield np.concatenate((pert_pose, shape, texture))

        for i in [2, 3]:  # tx, ty
            for pert in self.trans_perts:
                pert_pose = pose.copy()
                pert_pose[i] += pert
                yield np.concatenate((pert_pose, shape, texture))

        for i in xrange(len(shape)):
            for pert in self.param_perts:
                pert_shape = shape.copy()
                pert_shape[i] += pert * self.s_sigma[i] * scale
                yield np.concatenate((pose, pert_shape, texture))

        for i in xrange(len(texture)):
            for pert in self.param_perts:
                pert_texture = texture.copy()
                pert_texture[i] += pert * self.t_sigma[i]
                yield np.concatenate((pose, shape, pert_texture))
