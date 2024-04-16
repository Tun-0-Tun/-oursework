import cv2
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


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


image_path = "./dataset/Series015_body.tif"
image_path_matlab = "./dataset/Series015_1_slice_contour_matlab.tif"

binary_image = read_bin_image(image_path)
binary_image_matlab = read_bin_image(image_path_matlab)

binary_image = 1 - binary_image

plt.imshow(binary_image)
plt.show()
plt.imshow(binary_image_matlab)
plt.show()

print(binary_image_matlab.shape)

kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], np.uint8)

eroded_image = cv2.erode(binary_image, kernel, iterations=1)

contour = binary_image - eroded_image

overlay_img_pair(contour, binary_image_matlab)


non_zero_points = np.where(contour != 0)


for x, y in zip(non_zero_points[0], non_zero_points[1]):
    print("Координаты точки: ({}, {})".format(x, y))