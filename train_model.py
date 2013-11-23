#!/usr/bin/env python

import os
import argparse
from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel
from pyaam.patches import PatchesModel
from pyaam.texture import TextureModel
from pyaam.combined import CombinedModel
from pyaam.detector import FaceDetector



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', choices=['shape', 'patches', 'detector', 'texture', 'combined'], help='model name')
    parser.add_argument('--frac', type=float, default=0.99, help='fraction of variation')
    parser.add_argument('--kmax', type=int, default=20, help='maximum modes')
    parser.add_argument('--width', type=int, default=100, help='face width')
    parser.add_argument('--psize', type=int, default=11, help='patch size')
    parser.add_argument('--ssize', type=int, default=11, help='search window size')
    parser.add_argument('--var', type=float, default=1.0, help='variance of annotation error')
    parser.add_argument('-lmbda', type=float, default=1e-6, help='regularization weight')
    parser.add_argument('--mu', type=float, default=1e-3, help='initial stochastic gradient descent step size')
    parser.add_argument('--nsamples', type=int, default=1000, help='number of stochastic gradient descent samples')
    parser.add_argument('--face-width', type=int, default=100, help='face width')
    parser.add_argument('--shp-fn', default='data/shape.npz', help='shape model filename')
    parser.add_argument('--ptc-fn', default='data/patches.npz', help='patches model filename')
    parser.add_argument('--dtc-fn', default='data/detector.npz', help='face detector filename')
    parser.add_argument('--txt-fn', default='data/texture.npz', help='texture model filename')
    parser.add_argument('--cmb-fn', default='data/combined.npz', help='combined model filename')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    muct = MuctDataset()
    muct.load(clean=True)
    data = muct.all_lmks()
    imgs = muct.iterimages(mirror=True)
    print 'training samples:', len(data)

    if args.model == 'shape':
        print 'training shape model ...'
        model = ShapeModel.train(data.T, args.frac, args.kmax)
        print 'retained:', model.num_modes(), 'modes'
        model.save(args.shp_fn)
        print 'wrote', args.shp_fn

    elif args.model == 'patches':
        print 'reading images ...'
        imgs = list(imgs)
        print 'training patches model ...'
        sm = ShapeModel.load(args.shp_fn)
        model = PatchesModel.train(data.T, imgs, sm.get_shape(args.face_width), args.psize,
                                   args.ssize, args.var, args.lmbda, args.mu, args.nsamples)
        model.save(args.ptc_fn)
        print 'wrote', args.ptc_fn

    elif args.model == 'detector':
        print 'training face detector ...'
        sm = ShapeModel.load(args.shp_fn)
        model = FaceDetector.train(data.T, imgs, sm.get_shape())
        model.save(args.dtc_fn)
        print 'wrote', args.dtc_fn

    elif args.model == 'texture':
        print 'training texture model ...'
        sm = ShapeModel.load(args.shp_fn)
        ref = sm.get_shape(200, 150, 150)
        model = TextureModel.train(data.T, imgs, ref, args.frac, args.kmax)
        print 'retained:', model.num_modes(), 'modes'
        model.save(args.txt_fn)
        print 'wrote', args.txt_fn

    elif args.model == 'combined':
        print 'training combined model ...'
        sm = ShapeModel.load(args.shp_fn)
        ref = sm.get_shape(200, 150, 150)
        model = CombinedModel.train(data.T, imgs, ref, args.frac, args.kmax)
        print 'retained:', model.num_modes(), 'modes'
        model.save(args.cmb_fn)
        print 'wrote', args.cmb_fn
