#!/usr/bin/env python

import os
import argparse
from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frac', type=float, default=0.99, help='')
    parser.add_argument('--kmax', type=int, default=20, help='')
    parser.add_argument('--shp-fn', default='data/shape.npz', help='shape model filename')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    muct = MuctDataset()
    muct.load(clean=True)

    if not os.path.exists('data'):
        os.makedirs('data')

    data = muct.all_lmks()

    print 'shape model training samples:', len(data)
    smodel = ShapeModel.train(data.T, args.frac, args.kmax)
    print 'retained:', smodel.num_modes(), 'modes'
    smodel.save(args.shp_fn)
    print 'wrote', args.shp_fn

