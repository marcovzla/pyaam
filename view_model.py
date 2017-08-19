#!/usr/bin/env python

from __future__ import division

import sys
import cv2
import argparse
import numpy as np
from pyaam.shape import ShapeModel
from pyaam.patches import PatchesModel
from pyaam.texture import TextureModel
from pyaam.combined import CombinedModel
from pyaam.draw import Color, draw_string, draw_muct_shape, draw_texture
from pyaam.utils import get_vertices
from pyaam.texturemapper import TextureMapper



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', choices=['shape', 'patches', 'texture', 'combined'], help='model name')
    parser.add_argument('--scale', type=float, default=200, help='scale')
    parser.add_argument('--tranx', type=int, default=150, help='translate x')
    parser.add_argument('--trany', type=int, default=150, help='translate y')
    parser.add_argument('--width', type=int, default=300, help='image width')
    parser.add_argument('--height', type=int, default=300, help='image height')
    parser.add_argument('--face-width', type=int, default=200, help='face width')
    parser.add_argument('--shp-fn', default='data/shape.npz', help='shape model filename')
    parser.add_argument('--ptc-fn', default='data/patches.npz', help='patches model filename')
    parser.add_argument('--txt-fn', default='data/texture.npz', help='texture model filename')
    parser.add_argument('--cmb-fn', default='data/combined.npz', help='combined model filename')
    return parser.parse_args()



def genvals():
    """generate trajectory of parameters"""
    vals = np.empty(200)
    vals[:50] = np.arange(50) / 50
    vals[50:100] = (50 - np.arange(50)) / 50
    vals[100:] = -vals[:100]
    return vals



def view_shape_model(shp_fn, scale, tranx, trany, width, height):
    img = np.empty((height, width, 3), dtype='uint8')
    smodel = ShapeModel.load(shp_fn)
    vals = genvals()
    while True:
        for k in range(4, smodel.model.shape[1]):
            for v in vals:
                p = smodel.get_params(scale, tranx, trany)
                p[k] = p[0] * v * 3 * np.sqrt(smodel.variance[k])
                img[:] = 0  # set to black
                s = 'mode: %d, val: %f sd' % (k-3, v*3)
                draw_string(img, s)
                q = smodel.calc_shape(p)
                draw_muct_shape(img, q)
                cv2.imshow('shape model', img)
                if cv2.waitKey(10) & 0xff == 27:
                    sys.exit()



def view_texture_model(shp_fn, txt_fn, scale, tranx, trany, width, height):
    img = np.empty((height, width, 3), dtype='uint8')
    smodel = ShapeModel.load(shp_fn)
    tmodel = TextureModel.load(txt_fn)
    vals = genvals()
    # get reference shape
    ref = smodel.get_shape(scale, tranx, trany)
    ref = ref.reshape((ref.size//2, 2))
    while True:
        for k in range(tmodel.num_modes()):
            for v in vals:
                p = np.zeros(tmodel.num_modes())
                p[k] = v * 3 * np.sqrt(tmodel.variance[k])
                img[:] = 0
                s = 'mode: %d, val: %f sd' % (k, v*3)
                draw_string(img, s)
                t = tmodel.calc_texture(p)
                draw_texture(img, t, ref)
                cv2.imshow('texture model', img)
                if cv2.waitKey(10) & 0xff == 27:
                    sys.exit()



def view_combined_model(shp_fn, txt_fn, cmb_fn, scale, tranx, trany, width, height):
    img = np.empty((height, width, 3), dtype='uint8')
    cv2.namedWindow('combined model')
    tm = TextureMapper(img.shape[1], img.shape[0])
    smodel = ShapeModel.load(shp_fn)
    tmodel = TextureModel.load(txt_fn)
    cmodel = CombinedModel.load(cmb_fn)
    vals = genvals()
    params = smodel.get_params(scale, tranx, trany)
    ref = smodel.calc_shape(params)
    ref = ref.reshape((ref.size//2, 2))
    verts = get_vertices(ref)
    while True:
        for k in range(cmodel.num_modes()):
            for v in vals:
                p = np.zeros(cmodel.num_modes())
                p[k] = v * 3 * np.sqrt(cmodel.variance[k])
                sparams, tparams = cmodel.calc_shp_tex_params(p, smodel.num_modes())
                params[4:] = sparams

                shp = smodel.calc_shape(params)
                shp = shp.reshape(ref.shape)
                t = tmodel.calc_texture(tparams)
                img[:] = 0
                draw_texture(img, t, ref)
                warped = tm.warp_triangles(img, ref[verts], shp[verts])

                s = 'mode: %d, val: %f sd' % (k, v*3)
                draw_string(warped, s)
                cv2.imshow('combined model', warped)

                if cv2.waitKey(10) & 0xff == 27:
                    sys.exit()



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
    for i in range(len(pmodel.patches)):
        im = pmodel.patches[i]
        cv2.normalize(pmodel.patches[i], im, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        im = cv2.resize(im, (int(scale*im.shape[0]), int(scale*im.shape[1])))
        im = im.astype('uint8')
        patches.append(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
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
        cv2.rectangle(img, (x,y), (x+w, y+h), Color.red, 2, cv2.LINE_AA)
        text = 'patch %d' % (i+1)
        draw_string(img, text)
        cv2.imshow('patches model', img)
        c = cv2.waitKey(0) & 0xff
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

    elif args.model == 'texture':
        view_texture_model(args.shp_fn, args.txt_fn, 200, 150, 150, args.width, args.height)

    elif args.model == 'combined':
        view_combined_model(args.shp_fn, args.txt_fn, args.cmb_fn, 200, 150, 150, args.width, args.height)
