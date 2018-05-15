'''
preprocessing

Created on May 09 2018 11:48 
#@author: Kevin Le 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False
from dataset import SPCDataset

def main():
    pass

def aspect_resize(im):
    '''
    Preserve aspect ratio and resizes the image
    :param im: data array of image rescaled from 0->255
    :return: resized image
    '''
    ii = 256
    mm = [int (np.median (im[0, :, :])), int (np.median (im[1, :, :])), int (np.median (im[2, :, :]))]
    cen = np.floor (np.array ((ii, ii)) / 2.0).astype ('int')  # Center of the image
    dim = im.shape[0:2]
    if DEBUG:
        print("median {}".format (mm))
        print("ROC {}".format (cen))
        print("img dim {}".format (dim))
        # exit(0)

    if dim[0] != dim[1]:
        # get the largest dimension
        large_dim = max (dim)

        # ratio between the large dimension and required dimension
        rat = float (ii) / large_dim

        # get the smaller dimension that maintains the aspect ratio
        small_dim = int (min (dim) * rat)

        # get the indicies of the large and small dimensions
        large_ind = dim.index (max (dim))
        small_ind = dim.index (min (dim))
        dim = list (dim)

        # the dimension assigment may seem weird cause of how python indexes images
        dim[small_ind] = ii
        dim[large_ind] = small_dim
        dim = tuple (dim)
        if DEBUG:
            print('before resize {}'.format (im.shape))
        im = cv2.resize (im, dim)
        half = np.floor (np.array (im.shape[0:2]) / 2.0).astype ('int')
        if DEBUG:
            print('after resize {}'.format (im.shape))

        # make an empty array, and place the new image in the middle
        res = np.zeros ((ii, ii, 3), dtype='uint8')
        res[:, :, 0] = mm[0]
        res[:, :, 1] = mm[1]
        res[:, :, 2] = mm[2]

        if large_ind == 1:
            test = res[cen[0] - half[0]:cen[0] + half[0], cen[1] - half[1]:cen[1] + half[1] + 1]
            if test.shape != im.shape:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1] + 1] = im
            else:
                res[cen[0] - half[0]:cen[0] + half[0], cen[1] - half[1]:cen[1] + half[1] + 1] = im
        else:
            test = res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1]]
            if test.shape != im.shape:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1] + 1] = im
            else:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1]] = im
    else:
        res = cv2.resize (im, (ii, ii))
        half = np.floor (np.array (im.shape[0:2]) / 2.0).astype ('int')

    if DEBUG:
        print('aspect_resize: {}'.format (res.shape))
    return res

if __name__ == '__main__':
    def aspect_resize_tst():
        root = '/data6/lekevin/cayman'
        img_dir = '/data6/lekevin/cayman/rawdata'
        csv_filename = root + '/data/1/data_{}.csv'

        # Test initialization
        dataset = {phase: SPCDataset (csv_filename=csv_filename.format (phase), img_dir=img_dir, phase=phase) for phase
                   in ['train', 'val']}
        for phase in dataset:
            print (dataset[phase])

        # Test file, lbl retrieval
        fns, lbls = dataset['train'].get_fns ()
        testimg = fns[0]
        img = cv2.imread (testimg)
        img = (img * 255).astype (np.uint8)
        img = aspect_resize (img)
        plt.savefig('test.png', img)

    aspect_resize_tst()