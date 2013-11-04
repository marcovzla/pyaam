#!/usr/bin/env python

import os
import argparse
from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel
from pyaam.patches import PatchesModel
from pyaam.detector import FaceDetector



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', choices=['shape', 'patches', 'detector'], help='model name')
    parser.add_argument('--frac', type=float, default=0.99, help='')
    parser.add_argument('--kmax', type=int, default=20, help='')
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
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    muct = MuctDataset()
    muct.load(clean=True)
    data = muct.all_lmks()
    print 'training samples:', len(data)

    if args.model == 'shape':
        print 'training shape model ...'
        model = ShapeModel.train(data.T, args.frac, args.kmax)
        print 'retained:', model.num_modes(), 'modes'
        model.save(args.shp_fn)
        print 'wrote', args.shp_fn

    elif args.model == 'patches':
        print 'reading images ...'
        images = list(muct.iterimages(mirror=True))
        print 'training patches model ...'
        sm = ShapeModel.load(args.shp_fn)
        model = PatchesModel.train(data.T, images, sm.get_shape(args.face_width), args.psize,
                                   args.ssize, args.var, args.lmbda, args.mu, args.nsamples)
        model.save(args.ptc_fn)
        print 'wrote', args.ptc_fn

    elif args.model == 'detector':
        print 'training face detector ...'
        sm = ShapeModel.load(args.shp_fn)
        model = FaceDetector.train(data.T, muct.iterimages(mirror=True), sm.get_shape())
        model.save(args.dtc_fn)
        print 'wrote', args.dtc_fn
