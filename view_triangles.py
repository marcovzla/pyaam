#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import cv2
import argparse
from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel
from pyaam.draw import draw_polygons, Color
from pyaam.utils import get_aabb, normalize, warp_triangles, get_vertices



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-tm', dest='use_texturemapper', action='store_false',
                        help='disable TextureMapper')
    parser.add_argument('--no-norm', dest='normalize', action='store_false',
                        help='disable roi normalization')
    parser.add_argument('--no-trigs', dest='show_triangles', action='store_false',
                        help='hide triangles')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    cv2.namedWindow('triangles')
    cv2.namedWindow('warp')

    smodel = ShapeModel.load('data/shape.npz')
    params = smodel.get_params(200, 150, 150)
    ref = smodel.calc_shape(params)
    ref = ref.reshape((len(ref)//2, 2))

    verts = get_vertices(ref)

    muct = MuctDataset()
    muct.load(clean=True)

    if args.use_texturemapper:
        from pyaam.texturemapper import TextureMapper
        tm = TextureMapper(480, 640)
        warp_triangles = tm.warp_triangles

    for name, tag, lmks, flipped in muct.iterdata():
        img = muct.image(name)
        pts = lmks.reshape((len(lmks)//2, 2))
        aabb = get_aabb(pts)
        if args.normalize:
            img = normalize(img, aabb)
        orig = img.copy()
        if args.show_triangles:
            draw_polygons(img, pts[verts], Color.blue)
        cv2.imshow('triangles', img)
        warped = warp_triangles(orig, pts[verts], ref[verts])
        if args.show_triangles:
            draw_polygons(warped, ref[verts], Color.blue)
        cv2.imshow('warp', warped[:300,:300])
        key = cv2.waitKey()
        if key == 27:
            sys.exit()
