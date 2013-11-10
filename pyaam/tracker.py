# coding: utf-8

from __future__ import division

from pyaam.shape import ShapeModel
from pyaam.patches import PatchesModel
from pyaam.detector import FaceDetector

class FaceTracker(object):
    def __init__(self):
        self.shape = ShapeModel.load('data/shape.npz')
        self.patches = PatchesModel.load('data/patches.npz')
        self.detector = FaceDetector.load('data/detector.npz')
        self.points = None
        self.ssize = ((21,21), (11,11), (5,5))

    def reset(self):
        self.points = None

    def track(self, img):
        if self.points is None:
            self.points = self.detector.detect(img)
        for ssize in self.ssize:
            self.fit(img, ssize)

    def fit(self, img, ssize):
        p = self.shape.calc_params(self.points)
        pts = self.shape.calc_shape(p)
        peaks = self.patches.calc_peaks(img, pts, ssize)

        p = self.shape.calc_params(peaks)
        self.points = self.shape.calc_shape(p)
