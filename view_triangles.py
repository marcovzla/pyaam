#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import cv2
from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel
from pyaam.draw import draw_polygons, Color
from pyaam.utils import get_aabb, normalize, warp_triangles, get_vertices

if __name__ == '__main__':
    cv2.namedWindow('triangles')
    cv2.namedWindow('warp')

    smodel = ShapeModel.load('data/shape.npz')
    params = smodel.get_params(200, 150, 150)
    ref = smodel.calc_shape(params)
    ref = ref.reshape((len(ref)//2, 2))

    verts = get_vertices(ref)

    muct = MuctDataset()
    muct.load(clean=True)    

    for name, tag, lmks, flipped in muct.iterdata():
        img = muct.image(name)
        pts = lmks.reshape((len(lmks)//2, 2))
        aabb = get_aabb(pts)
        normalize(img, aabb)
        orig = img.copy()
        draw_polygons(img, pts[verts], Color.blue)
        cv2.imshow('triangles', img)
        warped = warp_triangles(orig, pts[verts], ref[verts])
        draw_polygons(warped, ref[verts], Color.blue)
        cv2.imshow('warp', warped[:300,:300])
        key = cv2.waitKey()
        if key == 27:
            sys.exit()
