#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import cv2
from view_model import draw_face
from pyaam.tracker import FaceTracker

if __name__ == '__main__':
    tracker = FaceTracker()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.exit('no camera')

    while True:
        val, img = cam.read()
        tracker.track(img)
        draw_face(img, tracker.points)
        cv2.imshow('face tracker', img)
        key = cv2.waitKey(10)
        if key == 27:  # esc
            break
        elif key == ord('r'):
            tracker.reset()
