#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import argparse
import numpy as np
import tables as tb
import cv2

from pyaam.muct import MuctDataset
from pyaam.draw import draw_polygons, draw_texture, draw_face
from pyaam.utils import get_vertices, get_aabb, normalize, get_mask
from pyaam.shape import ShapeModel
from pyaam.texture import TextureModel
from pyaam.texturemapper import TextureMapper
from pyaam.perturbator import Perturbator



def sample_texture(img, pts, ref, tm):
    """returns a texture vector"""
    img = normalize(img, get_aabb(pts))
    mask = get_mask(ref, img.shape[:2])
    verts = get_vertices(ref)
    warp = tm.warp_triangles(img, pts[verts], ref[verts])
    return warp[mask].ravel()



def experiments(images, landmarks, smodel, tmodel, ref_shape, fout):
    tm = TextureMapper(480, 640)
    tri = get_vertices(ref_shape)
    split = smodel.num_modes() + 4
    perturbator = Perturbator(np.sqrt(smodel.variance[4:]), np.sqrt(tmodel.variance))

    n_samples = len(landmarks)
    n_perts = perturbator.num_perts()
    total_perts = n_perts * n_samples
    n_params = 4 + smodel.num_modes() + tmodel.num_modes()
    t_vec_sz = tmodel.texture_vector_size()

    h5 = tb.openFile(fout, mode='w', title='perturbations')
    filters = tb.Filters(complevel=5, complib='blosc')
    P = h5.createCArray(h5.root, 'perturbations', tb.Float64Atom(),
            shape=(n_params, total_perts), filters=filters)
    R = h5.createCArray(h5.root, 'residuals', tb.Float64Atom(),
            shape=(t_vec_sz, total_perts), filters=filters,
            chunkshape=(2048, 128))

    for i in xrange(len(landmarks)):
        # get image and corresponding landmarks
        img = next(images)
        lmks = landmarks[i]
        pts = lmks.reshape(ref_shape.shape)
        # get shape and texture model parameters for current example
        s_params = smodel.calc_params(lmks)
        t_params = tmodel.calc_params(img, lmks, ref_shape, tm.warp_triangles)
        params = np.concatenate((s_params, t_params))

        perturbations = perturbator.perturbations(s_params, t_params)
        for j,pert in enumerate(perturbations):
            col = n_perts * i + j
            print 'perturbation {:,} of {:,}'.format(col+1, total_perts)
            s = pert[:split]
            t = pert[split:]
            x_image = smodel.calc_shape(s)
            x_image = x_image.reshape((x_image.size//2, 2))
            g_image = sample_texture(img, x_image, ref_shape, tm)
            g_model = tmodel.calc_texture(t)
            perturbation = pert - params
            residual = g_image - g_model

            P[:,col] = perturbation
            R[:,col] = residual

    h5.close()


def parse_args():
    description = ''  # FIXME write some description
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', dest='fout', default='data/perturbations.h5',
                        help='output file')
    parser.add_argument('--no-flip', action='store_false', dest='flipped',
                        help='exclude flipped data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    muct = MuctDataset()
    muct.load(clean=True)

    # If a face is too close to the image border and we perturbe
    # the scale then that face may grow beyond the image border.
    # Ignore problematic images.
    muct.ignore('i405wc-fn')

    data = muct.all_lmks()
    imgs = muct.iterimages(mirror=True)
    print 'training samples:', len(data)

    smodel = ShapeModel.load('data/shape.npz')
    tmodel = TextureModel.load('data/texture.npz')

    # get reference shape
    params = smodel.get_params(200, 150, 150)
    ref = smodel.calc_shape(params)
    ref = ref.reshape((ref.size//2, 2))

    experiments(imgs, data, smodel, tmodel, ref, args.fout)
    print 'wrote', args.fout
