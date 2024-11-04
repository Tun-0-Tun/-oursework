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

def plot_two_images(image1, image2, t2='MATLAB', t1='python'):
    # Найти общие минимальные и максимальные значения для цветовой шкалы
    common_min = min(image1.min(), image2.min())
    common_max = max(image1.max(), image2.max())

    # Создать фигуру с двумя графиками
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Отображение первого изображения с общей цветовой шкалой
    im1 = axes[0].imshow(image1, cmap='gray', vmin=common_min, vmax=common_max)
    axes[0].set_title(t1)
    axes[0].axis('off')  # Скрыть оси

    # Отображение второго изображения с общей цветовой шкалой
    im2 = axes[1].imshow(image2, cmap='gray', vmin=common_min, vmax=common_max)
    axes[1].set_title(t2)
    axes[1].axis('off')  # Скрыть оси

    # Добавить одну общую цветовую шкалу с настройкой размеров и отступов
    cbar = fig.colorbar(im1, ax= axes[1], orientation='vertical', shrink=0.8, pad=0.05)
    cbar.set_label('Values')

    # Отобразить график
    plt.tight_layout()
    plt.show()

def plot_image_difference(image1, image2, t1='python', t2='MATLAB'):
    # Вычислить модуль разности между двумя изображениями
    difference = np.abs(image1 - image2)

    # Найти минимальные и максимальные значения для цветовой шкалы
    common_min = min(difference.min(), image1.min(), image2.min())
    common_max = max(difference.max(), image1.max(), image2.max())

    # Создать фигуру с тремя графиками
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Отображение первого изображения
    im1 = axes[0].imshow(image1, cmap='gray', vmin=common_min, vmax=common_max)
    axes[0].set_title(t1)
    axes[0].axis('off')  # Скрыть оси

    # Отображение второго изображения
    im2 = axes[1].imshow(image2, cmap='gray', vmin=common_min, vmax=common_max)
    axes[1].set_title(t2)
    axes[1].axis('off')  # Скрыть оси

    # Отображение разности
    im_diff = axes[2].imshow(difference, cmap='hot', vmin=common_min, vmax=common_max)
    axes[2].set_title('Difference')
    axes[2].axis('off')  # Скрыть оси

    # Добавить общую цветовую шкалу для изображения разности
    cbar = fig.colorbar(im_diff, ax=axes[2], orientation='vertical', shrink=0.8, pad=0.05)
    cbar.set_label('Difference Values')

    # Отобразить график
    plt.tight_layout()
    plt.show()


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
    # plot_two_images(cellm2, cellm2)
    P2 = GetContour(cellB2, pntsN)

    print(f'Aligning rigidly. Overall {seqLength - 2} slices. Progress: ')
    for i in range(seqLength - 1):
        P1 = P2

        cellm_i = np.array(multi_layer_tiff[i+1], dtype=np.uint8)
        cellm_i[cellm_i == 0] = 255
        cellm_i[cellm_i == 1] = 0
        # plot_two_images(P1, P2)
        P2 = GetContour(cellm_i, pntsN)



        WP1, WP2 = match_contours(P1, P2, descTypeCorr, 'none', 0, np.pi / 2, 1.25, 1, 1)

        regParams, _ = cv2.estimateAffinePartial2D(WP2.astype(np.float32), WP1.astype(np.float32))
        Movl = np.dot(Movl, np.vstack([regParams, [0, 0, 1]]))
        MovlCur2First.append(Movl)

        newSliceM = cv2.warpAffine(cellm_i, Movl[:2:], (w, h))
        newSliceM = 255 - newSliceM
        new_tiff[i + 1,:,:]= newSliceM


        print(f'{i}/{seqLength - 2}')

    return icsNew, new_tiff, MovlCur2First





ics = []
pntsN =  500
descTypeCorr = "Centroid"
image_path = "./Series015_body.tif"
image_path_RA = "./Series015_RA_body.tif"


icsNew, new_tiff, MovlCur2First = ff_allign_rigid_ics_AO(ics, image_path, pntsN, descTypeCorr)
multi_layer_tiff = imageio.imread(image_path_RA)
cellm2 = np.array(multi_layer_tiff, dtype=np.uint8)
cellm2[cellm2 == 0] = 255
cellm2[cellm2 == 1] = 0

new_tiff = new_tiff.astype(int)
new_tiff[new_tiff == False] = 255
new_tiff[new_tiff == True] = 0

plot_image_difference(new_tiff[1], cellm2[1])

