#!/usr/bin/env python

from __future__ import division

import sys
import cv2
import numpy as np

from pyaam.muct import MuctDataset
from pyaam.draw import draw_face
from pyaam.utils import get_vertices
from pyaam.shape import ShapeModel
from pyaam.texture import TextureModel
from pyaam.detector import FaceDetector
from pyaam.texturemapper import TextureMapper



def test_aam(images, landmarks, smodel, tmodel, R, ref_shape):
    cv2.namedWindow('original')
    cv2.namedWindow('fitted')
    tm = TextureMapper(480, 640)
    tri = get_vertices(ref_shape)
    split = smodel.num_modes() + 4
    for i in xrange(len(landmarks)):
        img = next(images)
        cv2.imshow('original', img)
        lmks = landmarks[i].reshape(ref_shape.shape)
        pts = detector.detect(img)
        s_params = smodel.calc_params(pts)
        t_params = np.zeros(tmodel.num_modes())
        params = np.concatenate((s_params, t_params))

        shape = smodel.calc_shape(s_params)
        shape = shape.reshape((shape.size//2, 2))
        texture = tmodel.calc_texture(t_params)
        warped = draw_face(img, shape, texture, ref_shape, tm)
        cv2.imshow('fitted', warped)

        key = cv2.waitKey()
        if key == ord(' '):
            continue
        elif key == 27:
            sys.exit()



if __name__ == '__main__':
    smodel = ShapeModel.load('data/shape.npz')
    tmodel = TextureModel.load('data/texture.npz')
    detector = FaceDetector.load('data/detector.npz')
    R = np.load('data/regmat.npz')['R']

    muct = MuctDataset()
    muct.load(clean=True)
    data = muct.all_lmks()
    imgs = muct.iterimages(mirror=True)

    params = smodel.get_params(200, 150, 150)
    ref = smodel.calc_shape(params)
    ref = ref.reshape((ref.size//2, 2))

    test_aam(imgs, data, smodel, tmodel, R, ref)
