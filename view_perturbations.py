#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import argparse
import numpy as np
import cv2

from pyaam.muct import MuctDataset
from pyaam.draw import draw_polygons, draw_texture, draw_face
from pyaam.utils import get_vertices, sample_texture
from pyaam.shape import ShapeModel
from pyaam.texture import TextureModel
from pyaam.texturemapper import TextureMapper
from pyaam.perturbator import Perturbator



def experiments(images, landmarks, smodel, tmodel, ref_shape):
    cv2.namedWindow('original')
    cv2.namedWindow('model')
    cv2.namedWindow('perturbed')
    tm = TextureMapper(480, 640)
    tri = get_vertices(ref_shape)
    split = smodel.num_modes() + 4
    perturbator = Perturbator(np.sqrt(smodel.variance[4:]), np.sqrt(tmodel.variance))
    for i in xrange(len(landmarks)):
        # get image and corresponding landmarks
        img = next(images)
        lmks = landmarks[i]
        pts = lmks.reshape(ref_shape.shape)
        # get shape and texture model parameters for current example
        s_params = smodel.calc_params(lmks)
        t_params = tmodel.calc_params(img, lmks, ref_shape, tm)
        params = np.concatenate((s_params, t_params))

        cv2.imshow('original', img)

        shape = smodel.calc_shape(s_params)
        shape = shape.reshape((shape.size//2, 2))
        texture = tmodel.calc_texture(t_params)
        warped = draw_face(img, shape, texture, ref_shape, tm.warp_triangles)
        cv2.imshow('model', warped)

        for pert in perturbator.perturbations(s_params, t_params):
            s = pert[:split]
            t = pert[split:]
            x_image = smodel.calc_shape(s)
            x_image = x_image.reshape((x_image.size//2, 2))
            g_image = sample_texture(img, x_image, ref_shape, tm.warp_triangles)
            g_model = tmodel.calc_texture(t)
            perturbation = pert - params
            residual = g_image - g_model

            warped = draw_face(img, x_image, g_model, ref_shape, tm.warp_triangles)
            cv2.imshow('perturbed', warped)

            key = cv2.waitKey()
            if key == ord('n'):
                break
            elif key == 27:
                sys.exit()



def parse_args():
    description = ''  # FIXME write some description
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--no-flip', action='store_false', dest='flipped',
                        help='exclude flipped data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    muct = MuctDataset()
    muct.load(clean=True)
    data = muct.all_lmks()
    imgs = muct.iterimages(mirror=True)
    print 'training samples:', len(data)

    smodel = ShapeModel.load('data/shape.npz')
    tmodel = TextureModel.load('data/texture.npz')

    # get reference shape
    params = smodel.get_params(200, 150, 150)
    ref = smodel.calc_shape(params)
    ref = ref.reshape((ref.size//2, 2))

    experiments(imgs, data, smodel, tmodel, ref)
