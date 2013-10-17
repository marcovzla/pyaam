# coding: utf-8

from __future__ import division

import cv2
import cv2.cv as cv



class Color:
    black = cv.RGB(0, 0, 0)
    white = cv.RGB(255, 255, 255)
    red = cv.RGB(255, 0, 0)
    green = cv.RGB(0, 255, 0)
    blue = cv.RGB(0, 0, 255)
    cyan = cv.RGB(0, 255, 255)
    magenta = cv.RGB(255, 0, 255)
    yellow = cv.RGB(255, 255, 0)



def draw_string(img, text):
    font = cv2.FONT_HERSHEY_COMPLEX
    size, baseLine = cv2.getTextSize(text, font, 0.6, 1)
    cv2.putText(img, text, (0, size[1]), font, 0.6, Color.black, 1, cv2.CV_AA)
    cv2.putText(img, text, (1, size[1]+1), font, 0.6, Color.white, 1, cv2.CV_AA)

def draw_points(img, points, color):
    for p in points:
        cv2.circle(img, tuple(p), 2, color)

def draw_line(img, points, color):
    cv2.polylines(img, [points], False, color)

def draw_pairs(img, points, pairs, color):
    for a,b in pairs:
        cv2.line(img, tuple(points[a]), tuple(points[b]), color)
