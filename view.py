#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import cv2
import numpy as np
from pyaam.draw import Color, draw_points, draw_line, draw_pairs
from pyaam.muct import MuctDataset



class LmksDrawer(object):
    def __init__(self, img, img_flip, lmks, lmks_flip):
        self.img = img
        self.img_flip = img_flip
        self.lmks = lmks
        self.lmks_flip = lmks_flip

    def draw(self, flip=False, points=False, line=False, pairs=False):
        img = self.img_flip if flip else self.img
        lmks = self.lmks_flip if flip else self.lmks
        # draw on image copy
        img = img.copy()
        # prepare points
        pts = np.column_stack((lmks[::2], lmks[1::2]))
        # draw
        if line:
            draw_line(img, pts, Color.blue)
        if pairs:
            draw_pairs(img, pts, MuctDataset.PAIRS, Color.red)
        if points:
            draw_points(img, pts, Color.green)
        return img



if __name__ == '__main__':
    muct = MuctDataset()
    muct.load_clean()

    cv2.namedWindow('muct')
    flip = points = line = pairs = False

    for name, tag, lmks, flipped in muct.iterdata():
        print name
        drawer = LmksDrawer(muct.image(name), muct.image(name, True), lmks, flipped)

        while True:
            img = drawer.draw(flip, points, line, pairs) 
            cv2.imshow('muct', img)
            # handle keyboard events
            key = cv2.waitKey()
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
