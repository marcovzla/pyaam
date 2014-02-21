# coding: utf-8

from __future__ import division

import cv2
import numpy as np
from scipy.spatial import Delaunay



def get_aabb(pts):
    """axis-aligned minimum bounding box"""
    x, y = np.floor(pts.min(axis=0)).astype(int)
    w, h = np.ceil(pts.ptp(axis=0)).astype(int)
    return x, y, w, h

def get_vertices(pts):
    return Delaunay(pts).vertices

def normalize(img, aabb):
    x, y, w, h = aabb
    # work on image copy
    img = img.copy()
    # .copy() required on linux
    img[y:y+h,x:x+w,0] = cv2.equalizeHist(img[y:y+h,x:x+w,0].copy())
    img[y:y+h,x:x+w,1] = cv2.equalizeHist(img[y:y+h,x:x+w,1].copy())
    img[y:y+h,x:x+w,2] = cv2.equalizeHist(img[y:y+h,x:x+w,2].copy())
    return img

def get_mask(pts, shape):
    pts = pts.astype('int32')
    mask = np.zeros(shape, dtype='uint8')
    hull = cv2.convexHull(pts[:,np.newaxis].copy())  # .copy() required on linux
    hull = hull.reshape((hull.shape[0], hull.shape[2]))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask.astype(bool)

# NOTE you should use pyaam.texturemapper.TextureMapper instead
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

def pca(M, frac, kmax=None):
    """principal component analysis"""
    # see Stegmann's thesis section 5.6.1 for details
    enough_samples = M.shape[1] > M.shape[0]  # each column is a sample
    # covariance matrix
    C = M.dot(M.T) if enough_samples else M.T.dot(M)
    C /= M.shape[1]
    u, s, v = np.linalg.svd(C)
    if not enough_samples:
        u = M.dot(u)
    if kmax is not None:
        s = s[:kmax]
    p = s.cumsum() / s.sum()
    k = p[p < frac].size
    return u[:,:k]

def gram_schmid(V):
    """Gram-Schmid orthonormalization (in-place)"""
    n = V.shape[1]
    for i in xrange(n):
        for j in xrange(i):
            # subtract projection
            V[:,i] -= np.dot(V[:,i], V[:,j]) * V[:,j]
        # normalize
        V[:,i] /= np.linalg.norm(V[:,i])
    # orthonormalization was performed in-place
    # but we return V for convenience
    return V

def sample_texture(img, pts, ref, warp_triangles):
    """returns a texture vector"""
    aabb = get_aabb(pts)
    img = normalize(img, aabb)
    mask = get_mask(ref, img.shape[:2])
    verts = get_vertices(ref)
    warp = warp_triangles(img, pts[verts], ref[verts])
    return warp[mask].ravel()
