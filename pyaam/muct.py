# coding: utf-8

# The MUCT Face Database
# http://www.milbo.org/muct/

from __future__ import division

import os
import sys
import shutil
import urllib2
import tarfile
import numpy as np



# default dataset directory
DEFAULT_DATADIR = os.path.expanduser('~/pyaam_data/muct')



class MuctDataset(object):
    # dataset urls
    URLS = (
        'http://muct.googlecode.com/files/README.txt',
        'http://muct.googlecode.com/files/muct-landmarks-v1.tar.gz',
        'http://muct.googlecode.com/files/muct-a-jpg-v1.tar.gz',
        'http://muct.googlecode.com/files/muct-b-jpg-v1.tar.gz',
        'http://muct.googlecode.com/files/muct-c-jpg-v1.tar.gz',
        'http://muct.googlecode.com/files/muct-d-jpg-v1.tar.gz',
        'http://muct.googlecode.com/files/muct-e-jpg-v1.tar.gz',
    )

    def __init__(self, datadir=DEFAULT_DATADIR):
        self._datadir = datadir

    def download(self):
        """downloads and unpacks the muct dataset"""
        # delete datadir if it already exists
        if os.path.exists(self._datadir):
            shutil.rmtree(self._datadir)
        # create datadir
        os.makedirs(self._datadir)
        # change directory to datadir but don't forget where you came from
        cwd = os.getcwd()
        os.chdir(self._datadir)
        # download all files
        for url in self.URLS:
            filename = url.split('/')[-1]
            download(url, filename)
            # unpack file if needed
            if filename.endswith('.tar.gz'):
                with tarfile.open(filename) as tar:
                    tar.extractall()
        # return to original directory
        os.chdir(cwd)

    def load(self):
        # read landmarks file
        fname =  os.path.join(self._datadir, 'muct-landmarks/muct76-opencv.csv')
        data = np.loadtxt(fname, delimiter=',', skiprows=1, dtype=str)
        # separate data
        names = np.char.array(data[:,0])
        tags = data[:,1]
        landmarks = data[:,2:].astype(float)
        # find flipped data
        flipped = names.startswith('ir')
        # keep data in self
        self.names = names[~flipped]
        self.tags = tags[~flipped]
        self.landmarks = landmarks[~flipped]
        self.landmarks_flip = landmarks[flipped]

    def clean(self):
        """remove landmarks with unavailable points"""
        # unavailable points are marked with (0,0)
        is_complete = lambda x: all(x[::2] + x[1::2] != 0)
        keep = np.array(map(is_complete, self.landmarks))
        self.names = self.names[keep]
        self.tags = self.tags[keep]
        self.landmarks = self.landmarks[keep]
        self.landmarks_flip = self.landmarks_flip[keep]



# http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python/22776#22776
def download(url, fname):
    """downloads file and shows progress"""
    fsize_dl = 0
    block_sz = 8192
    u = urllib2.urlopen(url)
    with open(fname, 'wb') as f:
        meta = u.info()
        fsize = int(meta.getheaders('Content-Length')[0])
        sys.stdout.write('Downloading: %s Bytes: %s\n' % (fname, fsize))
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            f.write(buffer)
            fsize_dl += len(buffer)
            status = '%10d [%3.2f%%]\r' % (fsize_dl, fsize_dl * 100 / fsize)
            sys.stdout.write(status)
            sys.stdout.flush()
    # overwrite progress message
    sys.stdout.write(' ' * 20 + '\r')
    sys.stdout.flush()
