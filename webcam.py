#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import cv2

cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

cv2.namedWindow('webcam')
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    sys.exit('no camera')

while True:
    rval, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4,
                                     minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

    for x,y,w,h in rects:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

    cv2.imshow('webcam', img)

    if cv2.waitKey(5) & 0xff == 27:
        break
