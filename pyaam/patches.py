# coding: utf-8

from __future__ import division

import cv2
import cv2.cv as cv
import numpy as np



def prepare_image(img):
    gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    # gray = img.sum(axis=2) / img.shape[2]
    res = np.log(gray + 1)
    return res / res.max()

def make_gaussian(shape, var):
    """returns 2d gaussian of given shape and variance"""
    h,w = shape
    x = np.arange(w, dtype=float)
    y = np.arange(h, dtype=float)[:,np.newaxis]
    x0 = w // 2
    y0 = h // 2
    mat = np.exp(-0.5 * (pow(x-x0, 2) + pow(y-y0, 2)) / var)
    return cv2.normalize(mat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

def calc_response(img, patch, normalize=False):
    """calculates response map for image and patch"""
    img = prepare_image(img).astype('float32')
    patch = patch.astype('float32')
    res = cv2.matchTemplate(img, patch, cv2.TM_CCOEFF_NORMED)
    if normalize:
        res = cv2.normalize(res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        res /= res.sum()
    return res
