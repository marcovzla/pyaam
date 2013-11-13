# coding: utf-8

from __future__ import division

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.framebufferobjects import *



class TextureMapper(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # init glut and hide window
        glutInit()
        glutCreateWindow('TextureMapper')
        glutHideWindow()

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        glViewport(0, 0, width, height)

        glClearColor(0, 0, 0, 0)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_TEXTURE_2D)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def warp_triangles(self, image, src, dst, img_dst=None):
        width = self.width
        height = self.height

        dest_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, dest_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                0, GL_RGB, GL_UNSIGNED_BYTE, img_dst)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D, dest_texture, 0)

        orig_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, orig_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                0, GL_RGB, GL_UNSIGNED_BYTE, image)

        if img_dst is None:
            glClear(GL_COLOR_BUFFER_BIT)

        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -5, 5)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # assign texture to each triangle
        glBegin(GL_TRIANGLES)
        for s,d in zip(src, dst):
            glTexCoord2f(s[0,0]/width, s[0,1]/height)
            glVertex3f(d[0,0], d[0,1], 0)
            glTexCoord2f(s[1,0]/width, s[1,1]/height)
            glVertex3f(d[1,0], d[1,1], 0)
            glTexCoord2f(s[2,0]/width, s[2,1]/height)
            glVertex3f(d[2,0], d[2,1], 0)
        glEnd()

        # read from fbo
        s = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        warped = np.fromstring(s, dtype='uint8').reshape((height,width,3))

        glDeleteTextures(orig_texture)
        glDeleteTextures(dest_texture)
        return warped

    def cleanup(self):
        glDeleteFramebuffers(1, self.fbo)
