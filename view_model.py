#!/usr/bin/env python

from __future__ import division

import sys
import cv2
import argparse
import numpy as np
from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel
from pyaam.draw import Color, draw_string, draw_points, draw_pairs



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scale', type=float, default=200, help='scale')
    parser.add_argument('--tranx', type=int, default=150, help='translate x')
    parser.add_argument('--trany', type=int, default=150, help='translate y')
    parser.add_argument('--width', type=int, default=300, help='image width')
    parser.add_argument('--height', type=int, default=300, help='image height')
    parser.add_argument('--shp-fn', default='shape.npz', help='shape model filename')
    return parser.parse_args()



def genvals():
    """generate trajectory of parameters"""
    vals = np.empty(200)
    vals[:50] = np.arange(50) / 50
    vals[50:100] = (50 - np.arange(50)) / 50
    vals[100:] = -vals[:100]
    return vals



def view_shape_model(shp_fn, scale, tranx, trany, width, height):
    img = np.empty((height, width, 3), dtype='uint8')
    smodel = ShapeModel.load(shp_fn)
    vals = genvals()
    while True:
        for k in xrange(4, smodel.model.shape[1]):
            for v in vals:
                p = smodel.get_params(scale, tranx, trany)
                p[k] = p[0] * v * 3 * np.sqrt(smodel.variance[k])
                img[:] = 0  # set to black
                s = 'mode: %d, val: %f sd' % (k-3, v*3)
                draw_string(img, s)
                q = smodel.calc_shape(p)
                pts = np.column_stack((q[::2], q[1::2]))
                draw_pairs(img, pts, MuctDataset.PAIRS, Color.red)
                draw_points(img, pts, Color.green)
                cv2.imshow('shape model', img)
                if cv2.waitKey(10) == 27:
                    sys.exit()



if __name__ == '__main__':
    args = parse_args()
    view_shape_model(args.shp_fn, args.scale, args.tranx, args.trany, args.width, args.height)
