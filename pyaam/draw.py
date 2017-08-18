# coding: utf-8

from __future__ import division

import cv2
import numpy as np
from pyaam.muct import MuctDataset
from pyaam.utils import get_mask, get_vertices



class Color:
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    yellow = (255, 255, 0)



def prepare(x):
    return x.round().astype('int32')

def draw_string(img, text, font=cv2.FONT_HERSHEY_COMPLEX, scale=0.6, thickness=1):
    size, baseLine = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (0, size[1]), font, scale, Color.black, thickness, cv2.LINE_AA)
    cv2.putText(img, text, (1, size[1]+1), font, scale, Color.white, thickness, cv2.LINE_AA)

def draw_points(img, points, color, radius=2):
    points = prepare(points)
    for p in points:
        cv2.circle(img, tuple(p), radius, color)

def draw_line(img, points, color):
    points = prepare(points)
    cv2.polylines(img, [points], False, color)

def draw_pairs(img, points, pairs, color):
    points = prepare(points)
    for a,b in pairs:
        cv2.line(img, tuple(points[a]), tuple(points[b]), color)

def draw_polygons(img, polygons, color):
    polygons = prepare(polygons)
    for p in polygons:
        cv2.polylines(img, [p], True, color)

def draw_muct_shape(img, points):
    # convert vector of points into matrix of size (n_pts, 2)
    pts = points.reshape((len(points)//2, 2))
    draw_pairs(img, pts, MuctDataset.PAIRS, Color.red)
    draw_points(img, pts, Color.green)

def draw_texture(img, texture, points):
    texture = texture.round().reshape((texture.size//3, 3))
    mask = get_mask(points, img.shape[:2])
    img[mask] = texture
    return img

def draw_face(img, points, texture, ref_shape, warp_triangles):
    verts = get_vertices(ref_shape)
    img_texture = np.zeros(img.shape, dtype='uint8')
    draw_texture(img_texture, texture, ref_shape)
    return warp_triangles(img_texture, ref_shape[verts], points[verts], img)
