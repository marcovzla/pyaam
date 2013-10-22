#!/usr/bin/env python

from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel

if __name__ == '__main__':
    muct = MuctDataset()
    muct.load(clean=True)

    frac = 0.99
    kmax = 20
    filename = 'shape.npz'

    data = muct.all_lmks()

    print 'shape model training samples:', len(data)
    smodel = ShapeModel.train(data.T, frac, kmax)
    print 'retained:', smodel.num_modes(), 'modes'
    smodel.save(filename)
    print 'wrote', filename

