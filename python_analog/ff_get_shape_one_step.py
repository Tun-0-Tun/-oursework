import imageio
import numpy as np
from matplotlib import pyplot as plt
from GetContour import get_contour_img, GetContour
from algo_ff_built_FEM_LAME import algo_ff_built_FEM_LAME
from scipy.spatial import cKDTree

import scipy.io

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



def plot_two_contours(contour1, contour2, name1='Contour 1', name2='Contour 2'):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the first contour in blue
    ax.plot(contour1[:, 0], contour1[:, 1], color='blue', marker='o')

    # Plot the second contour in red
    ax.plot(contour2[:, 0], contour2[:, 1], color='red', marker='o')

    # Draw dotted lines connecting points with the same index
    for i in range(len(contour1)):
        ax.plot([contour1[i, 0], contour2[i, 0]], [contour1[i, 1], contour2[i, 1]], 'k:', lw=0.8)

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set the title and legend
    ax.set_title('Contours Plot')
    ax.legend([name1, name2])

    # Ensure aspect ratio is equal for proper scaling
    ax.set_aspect('equal', 'box')

    # Display the plot
    plt.show()


def GetFlowField(Pold, Pnew):
    ff = Pnew - Pold
    c = Pold
    return ff, c


def measure(image):
    # Находим индексы ненулевых элементов
    nonzero_indices = np.nonzero(image)

    # Находим минимальные и максимальные значения по каждой оси
    min_x = np.min(nonzero_indices[0])
    max_x = np.max(nonzero_indices[0])
    min_y = np.min(nonzero_indices[1])
    max_y = np.max(nonzero_indices[1])

    # Возвращаем координаты вершин минимального прямоугольника
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def MatchContours(contour1, contour2):
    tree = cKDTree(contour2)
    distances, indices = tree.query(contour1)
    nearest_points = contour2[indices]
    return contour1, nearest_points


def ff_get_shape_bckwrd_one_step(cellm1, cellm2, par):
    global ffX, ffY
    pntsN = par.get('pntsN', 80)
    intType = par.get('intType', 'TPS')
    matchType = par.get('matchType', 'DTW')
    verbose = par.get('verbose', 0)
    meshtype = par.get('meshtype', 'inner')

    # print(pntsN)

    if intType.upper() == 'FEM':
        useFEM = 2
    elif intType.upper() == 'TPS_GEO':
        useFEM = 1
    elif intType.upper() == 'TPS':
        useFEM = 0
    else:
        raise ValueError('Wrong interpolation type!')

    if matchType.upper() == 'DISTT':
        useDT = 1
    elif matchType.upper() == 'DTW':
        useDT = 0
    else:
        raise ValueError('Wrong matching type!')

    maxproj = np.sum(cellm, axis=0) > 0
    maxproj = (maxproj > 0).astype(int)
    m = measure(maxproj)[::2]
    ext_r = 5
    minX = max(0, m[0][1] - ext_r)
    minY = max(0, m[0][0] - ext_r)
    maxX = min(cellm.shape[2], m[1][1] + ext_r)
    maxY = min(cellm.shape[1], m[1][0] + ext_r)
    # cellm = cellm[:, minY:maxY, minX:maxX]
    flowField = []

    cellB1 = get_contour_img(cellm1)
    cellB2 = get_contour_img(cellm2)
    
    P1 = GetContour(cellB1, pntsN)
    P2 = GetContour(cellB2, pntsN)
    P1_not_sampled = GetContour(cellB1, 0)
    
    P1_matlab = scipy.io.loadmat('./Series015_RA_P1_P2.mat')['P1']
    P2_matlab = scipy.io.loadmat('./Series015_RA_P1_P2.mat')['P2']
    plot_two_contours(P1, P1_matlab, 'P1', 'P1_matlab')
    plot_two_contours(P2, P2_matlab, 'P2', 'P2_matlab')

    WP2, WP1 = MatchContours(P2, P1_not_sampled)

    WP1_matlab = scipy.io.loadmat('./Series015_RA_WP1_WP2.mat')['WP1']
    WP2_matlab = scipy.io.loadmat('./Series015_RA_WP1_WP2.mat')['WP2']
    plot_two_contours(WP1, WP1_matlab, 'WP1', 'WP1_matlab')
    plot_two_contours(WP2, WP2_matlab, 'WP2', 'WP2_matlab')


    # Для сравнения результатов
    ff, c = GetFlowField(WP2, WP1)
    # ff2, c2 = GetFlowField(WP2_matlab, WP1_matlab)
    flowField.append([c, ff])

    if verbose <= 2:
        if useFEM == 2:
            ffX, ffY = algo_ff_built_FEM_LAME(ff, c, np.array(cellB2.shape), par.get('Young', 100000),
                                              0.4, par.get('triHmax', 15), meshtype)
            # ffX, ffY = algo_ff_built_FEM_LAME(ff2, c2, np.array(cellB2.shape), par.get('Young', 100000),
            #                                   0.4, par.get('triHmax', 15), meshtype)

    return ffX, ffY, flowField


par = {
    'pntsN': 100,
    'intType': 'FEM',
    'matchType': 'DistT',
    'alphaDTW': 1.25,
    'resolveMultMatches': 1,
    'descTypeCorr': 'centroid',
    'descTypeDTW': 'centroid',
    'pntsScaleCorr': 1,
    'verbose': 0,
    'debugPath': '/path/to/debug',
    'ics': None,
    'meshtype': 'inner'
}

image_path = './Series015_RA_body.tif'
multi_layer_tiff = imageio.imread(image_path)
new_tiff = multi_layer_tiff.copy()
cellm = np.array(multi_layer_tiff, dtype=np.uint8)
cellm = cellm[:2:, :, :]
cellm[cellm == 0] = 255
cellm[cellm == 1] = 0
ffXics, ffYics, flowField = ff_get_shape_bckwrd_one_step(cellm[0], cellm[1], par)

ffXics_matlab = scipy.io.loadmat('./Series015_RA_ffX_new.mat')
ffYics_matlab =  scipy.io.loadmat('./Series015_RA_ffY_new.mat')

print(ffXics_matlab['ffX'].shape)
plot_image_difference(ffXics, ffXics_matlab['ffX'])
plot_image_difference(ffYics, ffYics_matlab['ffY'])