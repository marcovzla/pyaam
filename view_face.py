#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import cv2
import argparse
from pyaam.draw import draw_muct_shape
from pyaam.tracker import FaceTracker
from pyaam.detector import FaceDetector



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', choices=['tracker', 'detector'], help='model name')
    parser.add_argument('--detector', default='data/detector.npz', help='face detector filename')
    return parser.parse_args()



def view_face_tracker():
    tracker = FaceTracker()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.exit('no camera')
    while True:
        val, img = cam.read()
        tracker.track(img)
        draw_muct_shape(img, tracker.points)
        cv2.imshow('face tracker', img)
        key = cv2.waitKey(10)
        if key == 27:
            break
        elif key == ord('r'):
            tracker.reset()



def view_face_detector(detector_fn):
    detector = FaceDetector.load(detector_fn)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.exit('no camera')
    while True:
        val, img = cam.read()
        p = detector.detect(img)
        draw_muct_shape(img, p)
        cv2.imshow('face detector', img)
        if cv2.waitKey(10) == 27:
            break



if __name__ == '__main__':
    args = parse_args()

    if args.model == 'detector':
        view_face_detector(args.detector)

    elif args.model == 'tracker':
        view_face_tracker()
