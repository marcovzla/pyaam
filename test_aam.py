#!/usr/bin/env python

from __future__ import division

import sys
import cv2
import numpy as np

from pyaam.muct import MuctDataset
from pyaam.draw import draw_face, draw_muct_shape
from pyaam.utils import get_vertices
from pyaam.shape import ShapeModel
from pyaam.texture import TextureModel
from pyaam.detector import FaceDetector
from pyaam.texturemapper import TextureMapper
from pyaam.utils import sample_texture

MAX_ITER = 10


def get_instance(params, smodel, tmodel):
    split = smodel.num_params()
    s = params[:split]
    t = params[split:]
    shape = smodel.calc_shape(s)
    shape = shape.reshape((shape.size // 2, 2))
    texture = tmodel.calc_texture(t)
    return shape, texture



def test_aam(images, landmarks, smodel, tmodel, R, ref_shape):
    cv2.namedWindow('original')
    cv2.namedWindow('fitted')
    cv2.namedWindow('shape')
    tm = TextureMapper(480, 640)
    tri = get_vertices(ref_shape)
    for i in xrange(len(landmarks)):
        img = next(images)
        cv2.imshow('original', img)
        lmks = landmarks[i].reshape(ref_shape.shape)

        # detect face
        pts = detector.detect(img)
        # get params for detected face shape
        s_params = smodel.calc_params(pts)
        # mean texture
        t_params = np.zeros(tmodel.num_modes())
        # concatenate parameters
        params = np.concatenate((s_params, t_params))

        shape, texture = get_instance(params, smodel, tmodel)
        warped = draw_face(img, shape, texture, ref_shape, tm)
        cv2.imshow('fitted', warped)

        img2 = img.copy()
        draw_muct_shape(img2, shape.ravel())
        cv2.imshow('shape', img2)

        key = cv2.waitKey()
        if key == ord(' '):
            pass
        elif key == ord('n'):
            continue
        elif key == 27:
            sys.exit()

        shape, texture = get_instance(params, smodel, tmodel)
        g_image = sample_texture(img, shape, ref_shape, tm.warp_triangles)
        # compute residual
        residual = g_image - texture
        # evaluate error
        E0 = np.dot(residual, residual)
        # predict model displacements
        pert = R.dot(residual)

        for i in xrange(MAX_ITER):
            shape, texture = get_instance(params, smodel, tmodel)
            g_image = sample_texture(img, shape, ref_shape, tm.warp_triangles)
            # compute residual
            residual = g_image - texture
            # predict model displacements
            pert = R.dot(residual)
            for alpha in (1.5, 1, 0.5, 0.25, 0.125):
                new_params = params - alpha * pert

                shape, texture = get_instance(new_params, smodel, tmodel)
                g_image = sample_texture(img, shape, ref_shape, tm.warp_triangles)
                residual = g_image - texture
                Ek = np.dot(residual, residual)

                if Ek < E0:
                    params = new_params
                    E0 = Ek
                    break

            shape, texture = get_instance(params, smodel, tmodel)
            warped = draw_face(img, shape, texture, ref_shape, tm)
            cv2.imshow('fitted', warped)

            img2 = img.copy()
            draw_muct_shape(img2, shape.ravel())
            cv2.imshow('shape', img2)

            key = cv2.waitKey()
            if key == ord(' '):
                continue
            elif key == ord('n'):
                break
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
