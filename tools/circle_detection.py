'''
circle_detection

Created on May 22 2018 15:28 
#@author: Kevin Le 
'''
from __future__ import print_function, division

import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
from skimage import morphology, measure, exposure, restoration
from skimage.filters import threshold_otsu, scharr, gaussian
from skimage import img_as_ubyte
from scipy import ndimage
import cPickle as pickle

ROOT = '/data6/lekevin/cayman'
DEBUG=False

def main():
    print("Getting data...")
    data = get_fishegg_data()

    if DEBUG:
        idx = np.random.choice (data.index)
        image = data.loc[idx]['image']
        output_fn = os.path.join ('/data6/lekevin/cayman/records/model_d4/version_1/measurements',
                                  '/'.join (image.split ('/')[-2:]))
        dimA, dimB = detect_object_size (image, display=True, output_fn=output_fn)
        print (dimA, dimB)
        exit(0)

    measurements = {'image':[], 'A': [], 'B': []}

    cntr = 0
    for i, row in data.iterrows():
        image = row['image']
        output_fn = os.path.join ('/data6/lekevin/cayman/records/model_d4/version_1/measurements',
                                  '/'.join (image.split ('/')[-2:]))
        if not os.path.exists (os.path.dirname (output_fn)):
            os.makedirs (os.path.dirname (output_fn))
        try:
            dimA, dimB = detect_object_size (image, display=True, output_fn=output_fn)
            measurements['image'].append(image)
            measurements['A'].append (dimA)
            measurements['B'].append (dimB)
        except:
            print (image)
        print('{}/{} ({:.0f}%)     \r'.format(cntr, data.shape[0], 100.0*cntr/data.shape[0]), end='')
        cntr += 1
    # data['A'] = measurements['A']
    # data['B'] = measurements['B']
    pickle.dump(measurements, open('/data6/lekevin/cayman/records/model_d4/version_1/measurements.p', 'wb'))
    temp = pd.DataFrame(measurements)
    data = pd.merge(temp, data, on='image')

    # Clean up data
    data['image'] = data['image'].map (lambda x: '/'.join (str (x).split ('/')[-2:]))
    data = data[["image", "label", "A", "B", "human"]]
    data.to_csv('/data6/lekevin/cayman/records/model_d4/version_1/fish_egg.csv')
    print('Completed...')

def get_fishegg_data():
    # Grab labelled fish egg dataset and make into one dataframe
    df = pd.DataFrame()
    for i in range(3):  # Training Dataset 1
        if i == 2:    # Read in filename/label text files
            filename = os.path.join(ROOT, 'rawdata/yolksac.txt')
        else:
            filename = os.path.join(ROOT,'rawdata/classes_EC{}_1516combined.txt'.format(i+1))
        temp = pd.read_csv(filename, sep=' ', names=['image', 'label', 'cool_example'], header=None)
        temp['day'] = ['EC{}'.format(i+1)] * temp.shape[0]
        df = df.append(temp, ignore_index=True)

    with open (os.path.join (ROOT, 'rawdata/labels.txt')) as f:
        labels = {int (k): v for line in f for (k, v) in (line.strip ().split (None, 1),)}    # Map class labels to numeric labels
    df['class'] = df['label'].map(labels)

    pred_df = pd.DataFrame ()
    for i in range (2): # Training Dataset 4
        filename = os.path.join (ROOT, 'rawdata/d3_predictions{}.txt'.format (i))
        temp = pd.read_csv (filename, sep=',', names=['image', 'day', 'label'], header=None)
        pred_df = pred_df.append (temp, ignore_index=True)
    pred_df['day'] = pred_df['day'].map ({'Thu Feb 16': 'EC3', 'Wed Feb 15': 'EC2', 'Fri Feb 17': 'EC3'})
    df = df.append (pred_df)
    df['human'] = 1
    df = df.groupby(df['label']).get_group(1)

    path_map = {i: os.path.join (ROOT,'rawdata', '{}_SPC_Images_3-COLOR'.format (i)) for i in ['EC1', 'EC2', 'EC3']}
    df['image'] = df['day'].map (path_map).astype (str) + '/' + df['image']
    # Iterate and append

    csv_filename = '/data6/lekevin/cayman/records/model_d4/version_1/test_predictions.csv'
    temp = pd.read_csv (csv_filename)
    fish_egg = temp.groupby (temp['predictions']).get_group(1)
    fish_egg['human'] = 0
    fish_egg = fish_egg.rename(index=str, columns={'predictions':'label'})
    fish_egg = fish_egg[['image', 'label', 'human']]
    df = df.append(fish_egg, ignore_index=True)

    df.append(fish_egg, ignore_index=True)
    return df


def detect_object_size(filename, display=True, output_fn=None):
    orig_image = cv2.imread (filename)

    def midpoint(ptA, ptB):
        midPoint = (ptA + ptB) * 0.5
        return tuple (midPoint.astype (int))

    def preprocessing(img, display=False):
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur (img, (1, 1), 0, 0)
        img = cv2.Canny (img, threshold1=10, threshold2=100)
        img = cv2.dilate (img, None, iterations=1)
        img = cv2.erode (img, None, iterations=1)
        if display:
            axarr[1].imshow (img)
            axarr[1].set_title ('Canny Edge Detection')
        return img

    def preprocessing1(img, display=False):

        edge_thresh = 2.5
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        edges_mag = scharr (img)
        edges_med = np.median (edges_mag)
        edges_thresh = edge_thresh * edges_med
        edges = edges_mag >= edges_thresh
        edges = morphology.closing (edges, morphology.square (3))
        filled_edges = ndimage.binary_fill_holes (edges)
        edges = morphology.erosion (filled_edges, morphology.square (3))
        return edges

    # Detect edges & contours as preprocessing
    edges = preprocessing1 (orig_image, display=display)
    image = img_as_ubyte (edges)

    cnts = cv2.findContours (image.copy (), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2 () else cnts[1]
    (cnts, _) = contours.sort_contours (cnts)

    # Area & boxpoints then drawContours
    for i, c in enumerate (cnts):
        if cv2.contourArea (c) < 100:
            continue

        orig = orig_image.copy ()
        box = cv2.minAreaRect (c)
        box = cv2.boxPoints (box)
        box = np.array (box, dtype='int')
        cv2.drawContours (orig, [box], 0, (0, 0, 255), 1)

        box = perspective.order_points (box)

        (tl, tr, br, bl) = box
        tltr = midpoint (tl, tr)
        blbr = midpoint (bl, br)
        tlbl = midpoint (tl, bl)
        trbr = midpoint (tr, br)

        dA = dist.euclidean (tltr, blbr)
        dB = dist.euclidean (tlbl, trbr)

        cam_resolution = 3.1
        magnification = 0.137

        dimA = dA * cam_resolution / magnification / 1000
        dimB = dB * cam_resolution / magnification / 1000

        if display:
            fig, axarr = plt.subplots (1, 3, figsize=(15, 4))
            axarr[0].imshow (orig_image)
            axarr[0].set_title ('Original Image')
            axarr[1].imshow (edges)
            axarr[1].set_title ('Scharr Edge Detection')

            for (x, y) in box:
                cv2.circle (orig, (int (x), int (y)), 1, (0, 255, 0), 1)

            box = np.array ((tltr, blbr, tlbl, trbr))
            for (x, y) in box:
                cv2.circle (orig, (int (x), int (y)), 1, (0, 255, 0), 1)

            cv2.line (orig, tltr, blbr, (255, 0, 0), 1)
            cv2.line (orig, tlbl, trbr, (255, 0, 255), 1)
            cv2.putText (orig, "{:.2f}mm".format (dimA), (int (tltr[0] - 15), int (tltr[1] - 10)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 255), 1)
            cv2.putText (orig, "{:.2f}mm".format (dimB), (int (trbr[0] + 5), int (trbr[1])), cv2.FONT_HERSHEY_SIMPLEX,
                         0.22, (255, 255, 255), 1)

            axarr[2].imshow (orig)
            axarr[2].set_title ('measurmentA (red): {:.2f}mm\nmeasurmentB (pink): {:.2f}mm'.format (dimA, dimB),
                                fontsize=14)
            # plt.show ()

            if output_fn is not None:
                fig.savefig (output_fn)
		plt.close(fig)
        return dimA, dimB

if __name__ == '__main__':
    main()
