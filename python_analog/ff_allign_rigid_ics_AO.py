import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.transform import AffineTransform, warp
from match_contours import match_contours

from GetContour import get_contour_img, GetContour
import cv2
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


def ff_allign_rigid_ics_AO(ics, image_path, pntsN, descTypeCorr):
    multi_layer_tiff = imageio.imread(image_path)
    new_tiff = multi_layer_tiff.copy()
    cellm2 = np.array(multi_layer_tiff[0], dtype=np.uint8)
    cellm2[cellm2 == 0] = 255
    cellm2[cellm2 == 1] = 0
    w, h = cellm2.shape[1], cellm2.shape[0]

    icsNew = np.zeros_like(ics) if ics is not None else None
    seqLength = multi_layer_tiff.shape[0]
    Movl = np.eye(3)
    MovlCur2First = []

    cellB2 = get_contour_img(cellm2)
    P2 = GetContour(cellB2, pntsN)

    print(f'Aligning rigidly. Overall {seqLength - 2} slices. Progress: ')
    for i in range(seqLength - 1):
        P1 = P2

        cellm_i = np.array(multi_layer_tiff[i+1], dtype=np.uint8)
        cellm_i[cellm_i == 0] = 255
        cellm_i[cellm_i == 1] = 0
        P2 = GetContour(cellm_i, pntsN)

        WP1, WP2 = match_contours(P1, P2, descTypeCorr, 'none', 0, np.pi / 2, 1.25, 1, 1)

        regParams, _ = cv2.estimateAffinePartial2D(WP2.astype(np.float32), WP1.astype(np.float32))
        Movl = np.dot(Movl, np.vstack([regParams, [0, 0, 1]]))
        MovlCur2First.append(Movl)

        newSliceM = cv2.warpAffine(cellm_i, Movl[:2:], (w, h))
        newSliceM = 255 - newSliceM
        # plt.imshow(newSliceM, cmap='gray')
        # plt.colorbar()  # Добавление цветовой шкалы для отображения соответствия значений и цветов
        # plt.show()
        # newSliceM = warp(cellm[:, :, i + 1], tform.inverse, output_shape=(w, h))

        new_tiff[i + 1,:,:]= newSliceM


        print(f'{i}/{seqLength - 2}')

    return icsNew, new_tiff, MovlCur2First




# Пример вызова функции
ics = []  # Пример данных для ics (замените на реальные данные)
cellm = np.zeros((100, 100, 5))  # Пример данных для cellm (замените на реальные данные)
pntsN =  500 # Пример значения для pntsN (замените на реальное значение)
descTypeCorr = "Centroid"  # Пример значения для descTypeCorr (замените на реальное значение)
image_path = "./dataset/Series015_body.tif"
icsNew, new_tiff, MovlCur2First = ff_allign_rigid_ics_AO(ics, image_path, pntsN, descTypeCorr)

print(MovlCur2First)
save_image_path = "./res/Series015_ff.tif"
