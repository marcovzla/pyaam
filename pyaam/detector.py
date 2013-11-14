# coding: utf-8

from __future__ import division

import cv2
import cv2.cv as cv
import numpy as np



CASCADE_FILENAME = 'data/haarcascades/haarcascade_frontalface_default.xml'



class FaceDetector(object):
    def __init__(self, offset, ref):
        self.offset = offset
        self.ref_shape = ref
        self.cascade = cv2.CascadeClassifier(CASCADE_FILENAME)
        self.flags = cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv.CV_HAAR_SCALE_IMAGE

    @classmethod
    def train(cls, lmks, imgs, ref, frac=0.8, scale_factor=1.1, min_neighbors=2, min_size=(30,30)):
        detector = cv2.CascadeClassifier(CASCADE_FILENAME)
        flags = cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv.CV_HAAR_SCALE_IMAGE
        N = lmks.shape[1]
        xoffset = []
        yoffset = []
        zoffset = []

        for i in xrange(N):
            pts = lmks[:,i]
            img = next(imgs)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rects = detector.detectMultiScale(gray,
                                              scaleFactor=scale_factor,
                                              minNeighbors=min_neighbors,
                                              minSize=min_size,
                                              flags=flags)

            if len(rects) == 0:
                continue

            x,y,w,h = rects[0]
            if enough_bounded_points(pts, (x,y,w,h), frac):
                center = pts.reshape((pts.size//2,2)).sum(axis=0) / (pts.size//2)
                xoffset.append((center[0] - (x + 0.5 * w)) / w)
                yoffset.append((center[1] - (y + 0.5 * h)) / w)
                zoffset.append(calc_scale(pts, ref) / w)

        xoffset.sort()
        yoffset.sort()
        zoffset.sort()

        detector_offset = (xoffset[len(xoffset)//2],
                           yoffset[len(yoffset)//2],
                           zoffset[len(zoffset)//2])

        return cls(detector_offset, ref)

    @classmethod
    def load(cls, filename):
        arch = np.load(filename)
        return cls(arch['offset'], arch['ref_shape'])

    def save(self, filename):
        np.savez(filename, offset=self.offset, ref_shape=self.ref_shape)

    def detect(self, img, scale_factor=1.1, min_neighbors=2, min_size=(30,30)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = self.cascade.detectMultiScale(gray,
                                              scaleFactor=scale_factor,
                                              minNeighbors=min_neighbors,
                                              minSize=min_size,
                                              flags=self.flags)

        if len(rects) == 0:
            return

        r = rects[0]
        scale = self.offset * r[2]
        p = self.ref_shape.copy()
        p[::2] = scale[2] * p[::2] + r[0] + 0.5 * r[2] + scale[0]
        p[1::2] = scale[2] * p[1::2] + r[1] + 0.5 * r[3] + scale[1]
        return p



def calc_scale(pts, ref):
    pts = pts.reshape((pts.size//2,2))
    c = pts.sum(axis=0) / pts.shape[0]
    p = pts - c
    p = p.reshape(p.size)
    ref = ref.reshape(ref.size)
    return ref.dot(p) / ref.dot(ref)

def enough_bounded_points(pts, r, frac):
    n = len(pts) // 2
    bounded = pts[::2] >= r[0]
    bounded &= pts[::2] <= r[0]+r[2]
    bounded &= pts[1::2] >= r[1]
    bounded &= pts[1::2] <= r[1]+r[3]
    return bounded.sum() / n >= frac

