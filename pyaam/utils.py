# coding: utf-8

from __future__ import division

import cv2
import numpy as np
from scipy.spatial import Delaunay



def get_aabb(pts):
    x, y = np.floor(pts.min(axis=0)).astype(int)
    w, h = np.ceil(pts.ptp(axis=0)).astype(int)
    return x, y, w, h

def get_vertices(pts):
    return Delaunay(pts).vertices

def normalize(img, aabb):
    x, y, w, h = aabb
    img[y:y+h,x:x+w,0] = cv2.equalizeHist(img[y:y+h,x:x+w,0])
    img[y:y+h,x:x+w,1] = cv2.equalizeHist(img[y:y+h,x:x+w,1])
    img[y:y+h,x:x+w,2] = cv2.equalizeHist(img[y:y+h,x:x+w,2])
    return img

def warp_triangles(img, src, dst):
    result = np.zeros(img.shape, dtype='uint8')
    dsize = (img.shape[1], img.shape[0])
    for s, d in zip(src, dst):
        mask = np.zeros(img.shape[:2], dtype='uint8')
        cv2.fillConvexPoly(mask, d.astype('int32'), 255)
        mask = mask.astype(bool)
        M = cv2.getAffineTransform(s.astype('float32'), d.astype('float32'))
        warp = cv2.warpAffine(img, M, dsize, flags=cv2.INTER_LINEAR)
        result[mask] = warp[mask]
    return result

