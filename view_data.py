#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import cv2
import numpy as np
from pyaam.draw import Color, draw_string, draw_points, draw_line, draw_pairs
from pyaam.muct import MuctDataset



class LandmarkDrawer(object):
    def __init__(self, name, img, img_flip, lmks, lmks_flip):
        self.name = name
        self.name_flip = name[0] + 'r' + name[1:]  # using muct naming scheme
        self.img = img
        self.img_flip = img_flip
        self.lmks = lmks
        self.lmks_flip = lmks_flip

    def draw(self, flip=False, points=False, line=False, pairs=False):
        name = self.name_flip if flip else self.name
        img = self.img_flip if flip else self.img
        lmks = self.lmks_flip if flip else self.lmks
        # draw on image copy
        img = img.copy()
        # prepare points
        pts = np.column_stack((lmks[::2], lmks[1::2]))
        # draw
        draw_string(img, name)
        if line: draw_line(img, pts, Color.blue)
        if pairs: draw_pairs(img, pts, MuctDataset.PAIRS, Color.red)
        if points: draw_points(img, pts, Color.green)
        return img



if __name__ == '__main__':
    muct = MuctDataset()
    muct.load(clean=True)

    cv2.namedWindow('muct')
    flip = points = line = pairs = False

    for name, tag, lmks, flipped in muct.iterdata():
        image = muct.image(name)
        image_flip = muct.image(name, flip=True)
        drawer = LandmarkDrawer(name, image, image_flip, lmks, flipped)

        while True:
            img = drawer.draw(flip, points, line, pairs) 
            cv2.imshow('muct', img)
            # handle keyboard events
            key = cv2.waitKey() &0xff
            if key == 27:
                sys.exit()
            elif key == ord(' '):
                break  # next image
            elif key == ord('1'):
                flip = not flip
            elif key == ord('2'):
                points = not points
            elif key == ord('3'):
                line = not line
            elif key == ord('4'):
                pairs = not pairs
