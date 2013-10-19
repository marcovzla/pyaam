#!/usr/bin/env python

from __future__ import division

import sys
import cv2
import numpy as np
from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel
from pyaam.draw import Color, draw_string, draw_points, draw_pairs

if __name__ == '__main__':
    fname = 'shape.npz'
    scale = 200
    tranx = 150
    trany = 150
    width = 300
    height = 300

    smodel = ShapeModel.load(fname)

    img = np.empty((height, width, 3), dtype='uint8')

    vals = np.empty(200)
    vals[:50] = np.arange(50) / 50
    vals[50:100] = (50 - np.arange(50)) / 50
    vals[100:] = -vals[:100]

    while True:
        for k in xrange(4, smodel.model.shape[1]):
            for v in vals:
                p = smodel.get_params(scale, tranx, trany)
                p[k] = p[0] * v * 3 * np.sqrt(smodel.variance[k])

                img[:] = 0

                s = 'mode: %d, val: %f sd' % (k-3, v*3)
                draw_string(img, s)

                q = smodel.calc_shape(p)

                pts = np.column_stack((q[::2], q[1::2]))
                pts = pts.round().astype('int32')

                draw_pairs(img, pts, MuctDataset.PAIRS, Color.red)
                draw_points(img, pts, Color.green)

                cv2.imshow('shape model', img)
                if cv2.waitKey(10) == 27:
                    sys.exit()
