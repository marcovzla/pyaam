#!/usr/bin/env python

from __future__ import division

import sys
import cv2
import cv2.cv as cv
import argparse
import numpy as np
from pyaam.muct import MuctDataset
from pyaam.shape import ShapeModel
from pyaam.patches import PatchesModel
from pyaam.detector import FaceDetector
from pyaam.draw import Color, draw_string, draw_points, draw_pairs



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', choices=['shape', 'patches', 'detector'], help='model name')
    parser.add_argument('--scale', type=float, default=200, help='scale')
    parser.add_argument('--tranx', type=int, default=150, help='translate x')
    parser.add_argument('--trany', type=int, default=150, help='translate y')
    parser.add_argument('--width', type=int, default=300, help='image width')
    parser.add_argument('--height', type=int, default=300, help='image height')
    parser.add_argument('--face-width', type=int, default=200, help='face width')
    parser.add_argument('--shp-fn', default='data/shape.npz', help='shape model filename')
    parser.add_argument('--ptc-fn', default='data/patches.npz', help='patches model filename')
    parser.add_argument('--dtc-fn', default='data/detector.npz', help='face detector filename')
    return parser.parse_args()



def genvals():
    """generate trajectory of parameters"""
    vals = np.empty(200)
    vals[:50] = np.arange(50) / 50
    vals[50:100] = (50 - np.arange(50)) / 50
    vals[100:] = -vals[:100]
    return vals



def draw_face(img, points):
    # convert vector of points into matrix of size (n_pts, 2)
    pts = points.reshape((len(points)//2, 2))
    draw_pairs(img, pts, MuctDataset.PAIRS, Color.red)
    draw_points(img, pts, Color.green)



def view_shape_model(shp_fn, scale, tranx, trany, width, height):
    img = np.empty((height, width, 3), dtype='uint8')
    smodel = ShapeModel.load(shp_fn)
    vals = genvals()
    while True:
        for k in xrange(4, smodel.model.shape[1]):
            for v in vals:
                p = smodel.get_params(scale, tranx, trany)
                p[k] = p[0] * v * 3 * np.sqrt(smodel.variance[k])
                img[:] = 0  # set to black
                s = 'mode: %d, val: %f sd' % (k-3, v*3)
                draw_string(img, s)
                q = smodel.calc_shape(p)
                draw_face(img, q)
                cv2.imshow('shape model', img)
                if cv2.waitKey(10) == 27:
                    sys.exit()



def view_face_detector(dtc_fn):
    detector = FaceDetector.load(dtc_fn)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.exit('no camera')
    while True:
        val, img = cam.read()
        p = detector.detect(img)
        draw_face(img, p)
        cv2.imshow('face detector', img)
        if cv2.waitKey(10) == 27:
            break



def view_patches_model(ptc_fn, shp_fn, width):
    pmodel = PatchesModel.load(ptc_fn)
    smodel = ShapeModel.load(shp_fn)
    ref = pmodel.ref_shape
    ref = np.column_stack((ref[::2], ref[1::2]))
    # compute scale factor
    scale = width / ref[:,0].ptp()
    height = int(scale * ref[:,1].ptp() + 0.5)
    # compute image width
    max_height = int(scale * pmodel.patches.shape[1])
    max_width = int(scale * pmodel.patches.shape[2])
    # create reference image
    image_size = (height+4*max_height, width+4*max_width, 3)
    image = np.empty(image_size, dtype='uint8')
    image[:] = 192
    patches = []
    points = []
    for i in xrange(len(pmodel.patches)):
        im = cv2.normalize(pmodel.patches[i], alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        im = cv2.resize(im, (int(scale*im.shape[0]), int(scale*im.shape[1])))
        im = im.astype('uint8')
        patches.append(cv2.cvtColor(im, cv.CV_GRAY2BGR))
        h,w = patches[i].shape[:2]
        points.append((int(scale*ref[i,1] + image_size[0]/2 - h/2),
                       int(scale*ref[i,0] + image_size[1]/2 - w/2)))
        y,x = points[i]
        image[y:y+h,x:x+w,:] = patches[i]
    cv2.namedWindow('patches model')
    i = 0
    while True:
        img = image.copy()
        y,x = points[i]
        h,w = patches[i].shape[:2]
        img[y:y+h,x:x+w,:] = patches[i]  # draw current patch on top
        cv2.rectangle(img, (x,y), (x+w, y+h), Color.red, 2, cv2.CV_AA)
        text = 'patch %d' % (i+1)
        draw_string(img, text)
        cv2.imshow('patches model', img)
        c = cv2.waitKey(0)
        if c == 27:
            break
        elif c == ord('j'):
            i += 1
        elif c == ord('k'):
            i -= 1
        if i < 0:
            i = 0
        elif i >= len(pmodel.patches):
            i = len(pmodel.patches) - 1



if __name__ == '__main__':
    args = parse_args()

    if args.model == 'shape':
        view_shape_model(args.shp_fn, args.scale, args.tranx, args.trany,
                         args.width, args.height)

    elif args.model == 'patches':
        view_patches_model(args.ptc_fn, args.shp_fn, args.face_width)

    elif args.model == 'detector':
        view_face_detector(args.dtc_fn)
