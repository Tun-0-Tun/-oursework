import numpy as np
from scipy.interpolate import interp1d
from skimage import measure
import cv2
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import chain_code as cc
import diplib as dip

from ResampleContourPoints2D import ResampleContourPoints2D

def overlay_img_pair(im1, im2):
    if im1.shape != im2.shape:
        raise "im1 size must be equal to im2 size"
    overlay_img = np.dstack((im1, im2, np.zeros(im1.shape)))
    plt.imshow(overlay_img)
    plt.show()


def read_bin_image(image_path):
    multi_layer_tiff = imageio.imread(image_path)
    if len(multi_layer_tiff.shape) == 3:
        return np.array(multi_layer_tiff[0], dtype=np.uint8)
    else:
        return np.array(multi_layer_tiff, dtype=np.uint8)


# image_path = "./dataset/Series015_body.tif"
# image_path_matlab = "./dataset/Series015_1_slice_contour_matlab.tif"

def skeletonize(binary_image):
    dip_binary_image = dip.Image(binary_image) > 0
    skel = dip.EuclideanSkeleton(dip_binary_image)
    return np.asarray(skel)

def get_contour_img(binary_image):
    binary_image = 1 - binary_image
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    contour = binary_image - eroded_image
    return skeletonize(contour)


def GetContour(contourIm, pntsNum=60, step=None):
    hain_code, contour = cc.trace_boundary(contourIm, 8, 0)
    contour = np.array(contour)
    contour = contour[:, ::-1]
    contour = contour[::-1]
    contour = contour[2::]
    y = contour[:, 1] + 1
    x = contour[:, 0] + 1
    xc = np.mean(x)
    yc = np.mean(y)
    p, a = np.arctan2(y - yc, x - xc), np.hypot(x - xc, y - yc)

    if (p[1] - p[0] < 0 and p[0] * p[1] > 0) or (p[0] * p[1] < 0 and p[0] < 0):
        p = np.flip(p)
        a = np.flip(a)

    ind = np.where((p > 0) & (p < 0.1))[0][0]
    p = np.roll(p, -ind)
    a = np.roll(a, -ind)
    x, y = a * np.cos(p), a * np.sin(p)
    x += xc
    y += yc
    P = np.array([x, y]).T

    if pntsNum > 0:
        return ResampleContourPoints2D(P, pntsNum)
    else:
        return P

def get_contour_seq(cellm):
    cellm_contour = np.zeros_like(cellm).astype(float)
    for i in range(len(cellm)):
        cellm_contour[i] = get_contour_img(cellm[i])
    return cellm_contour

