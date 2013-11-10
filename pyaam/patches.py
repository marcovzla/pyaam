# coding: utf-8

from __future__ import division

import cv2
import cv2.cv as cv
import numpy as np



class PatchesModel(object):
    def __init__(self, patches, ref_shape):
        self.patches = patches
        self.ref_shape = ref_shape

    @classmethod
    def train(cls, lmks, imgs, ref, psize, ssize, var, lmbda, mu_init, nsamples):
        patches = train_patches(lmks, imgs, ref, psize, ssize, var, lmbda, mu_init, nsamples)
        return cls(patches, ref)

    @classmethod
    def load(cls, filename):
        arch = np.load(filename)
        return cls(arch['patches'], arch['ref_shape'])

    def save(self, filename):
        np.savez(filename, patches=self.patches, ref_shape=self.ref_shape)

    def calc_peaks(self, img, points, ssize):
        points = points.reshape((len(points)//2, 2))
        return calc_peaks(img, points, ssize, self.patches, self.ref_shape).flatten()



def train_patch(images, psize, var=1.0, lmbda=1e-6, mu_init=1e-3, nsamples=1000):
    """
    images - featured centered training images
    psize - desired patch model size
    var - variance of annotation error
    lmbda - regularization parameter
    mu_init - initial step size
    nsamples - number of stochastic samples
    """

    h,w = psize
    N = len(images)
    n = w * h
    yimg, ximg = images[0].shape[:2]
    dx = ximg - w
    dy = yimg - h

    # ideal response map
    F = make_gaussian((dy,dx), var)

    # allocate memory
    dP = np.empty(psize)
    O = np.ones(psize) / n
    P = np.zeros(psize)

    # stochastic gradient descent
    mu = mu_init
    step = pow(1e-8/mu_init, 1/nsamples)
    for sample in xrange(nsamples):
        i = np.random.randint(N)
        I = convert_image(images[i])
        dP[:] = 0
        # compute gradient direction
        for y in xrange(dy):
            for x in xrange(dx):
                Wi = I[y:y+h,x:x+w].copy()
                Wi -= Wi.dot(O)
                Wi = cv2.normalize(Wi)
                dP += (F[y,x] - P.dot(Wi)) * Wi
        # take a small step
        P += mu * (dP - lmbda*P)
        # reduce step size
        mu *= step

    return P

def train_patches(lmks, imgs, ref, psize, ssize, var=1.0, lmbda=1e-6, mu_init=1e-3, nsamples=1000):
    """
    ref - reference shape
    psize - desired patch size
    ssize - search window size
    var - variance of annotation error
    lmbda - regularization weight
    mu_init - initial stochastic gradient descent step size
    nsamples - number of stochastic gradient descent samples
    """

    if isinstance(psize, int):
        psize = (psize, psize)
    if isinstance(ssize, int):
        ssize = (ssize, ssize)

    n = len(ref) // 2
    ximg = psize[1] + ssize[1]
    yimg = psize[0] + ssize[0]
    wsize = (yimg, ximg)

    patches = []

    # train each patch model
    for i in xrange(n):
        print 'patch', i+1, 'of', n, '...'
        images = []
        for j in xrange(lmks.shape[1]):
            im = imgs[j]
            pt = lmks[:,j]
            S = calc_simil(pt, ref)
            A = np.empty((2,3))
            A[:2,:2] = S[:2,:2]
            A[0,2] = pt[2*i] - (A[0,0] * (ximg-1)/2 + A[0,1] * (yimg-1)/2)
            A[1,2] = pt[2*i+1] - (A[1,0] * (ximg-1)/2 + A[1,1] * (yimg-1)/2)
            I = cv2.warpAffine(im, A, wsize, flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
            images.append(I)

        patches.append(train_patch(images, psize, var, lmbda, mu_init, nsamples))

    return np.array(patches)

def calc_simil(pts, ref):
    # compute translation
    n = len(pts) // 2
    mx = pts[::2].sum() / n
    my = pts[1::2].sum() / n
    p = np.empty(pts.shape)
    p[::2] = pts[::2] - mx
    p[1::2] = pts[1::2] - my
    # compute rotation and scale
    a = np.sum(ref[::2] ** 2 + ref[1::2] ** 2)
    b = np.sum(ref[::2] * p[::2] + ref[1::2] * p[1::2])
    c = np.sum(ref[::2] * p[1::2] + ref[1::2] * p[::2])
    b /= a
    c /= a
    scale = np.sqrt(b ** 2 + c ** 2)
    theta = np.arctan2(c,b)
    sc = scale * np.cos(theta)
    ss = scale * np.sin(theta)
    return np.array([[sc,-ss,mx],[ss,sc,my]])

def make_gaussian(shape, var):
    """returns 2d gaussian of given shape and variance"""
    h,w = shape
    x = np.arange(w, dtype=float)
    y = np.arange(h, dtype=float)[:,np.newaxis]
    x0 = w // 2
    y0 = h // 2
    mat = np.exp(-0.5 * (pow(x-x0, 2) + pow(y-y0, 2)) / var)
    return cv2.normalize(mat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

def convert_image(img):
    gray = img.sum(axis=2) / img.shape[2]
    res = np.log(gray + 1)
    return res / res.max()

def calc_response(img, patch, normalize=False):
    """calculates response map for image and patch"""
    img = convert_image(img).astype('float32')
    patch = patch.astype('float32')
    res = cv2.matchTemplate(img, patch, cv2.TM_CCOEFF_NORMED)
    if normalize:
        res = cv2.normalize(res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        res /= res.sum()
    return res

def apply_simil(S, pts):
    p = np.empty(pts.shape)
    p[:,0] = S[0,0] * pts[:,0] + S[0,1] * pts[:,1] + S[0,2]
    p[:,1] = S[1,0] * pts[:,0] + S[1,1] * pts[:,1] + S[1,2]
    return p

def inv_simil(S):
    Si = np.empty((2,3))
    d = S[0,0] * S[1,1] - S[1,0] * S[0,1]
    Si[0,0] = S[1,1] / d
    Si[0,1] = -S[0,1] / d
    Si[1,0] = -S[1,0] / d
    Si[1,1] = S[0,0] / d
    Si[:,2] = -Si[:,:2].dot(S[:,2])
    return Si

def calc_peaks(img, points, ssize, patches, ref):
    assert len(points) == len(patches)
    pt = points.flatten()
    S = calc_simil(pt, ref)
    Si = inv_simil(S)
    pts = apply_simil(Si, points)
    for i in xrange(len(points)):
        patch = patches[i]
        psize = patch.shape
        wsize = (ssize[0]+psize[0],ssize[1]+psize[1])
        A = np.empty((2,3))
        A[:,:2] = S[:,:2]
        A[:,2] = points[i,:] - (A[:,0] * (wsize[1]-1)/2 + A[:,1] * (wsize[0]-1)/2)
        I = cv2.warpAffine(img, A, wsize, flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
        R = calc_response(I, patch)
        maxloc = cv2.minMaxLoc(R)[-1]
        pts[i,0] = pts[i,0] + maxloc[0] - 0.5 * ssize[0]
        pts[i,1] = pts[i,1] + maxloc[1] - 0.5 * ssize[1]
    return apply_simil(S, pts)
