import imageio
import numpy as np
from matplotlib import pyplot as plt
from GetContour import get_contour_img, GetContour
from algo_ff_built_FEM_LAME import algo_ff_built_FEM_LAME
from scipy.spatial import cKDTree

import scipy.io

def plot_two_images(image1, image2):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image with its original range for the colorbar
    im1 = axes[0].imshow(image1, cmap='gray', vmin=image1.min(), vmax=image1.max())
    axes[0].set_title('Image 1')
    axes[0].axis('off')  # Hide axis

    # Display the second image with its original range for the colorbar
    im2 = axes[1].imshow(image2, cmap='gray', vmin=image2.min(), vmax=image2.max())
    axes[1].set_title('Image 2')
    axes[1].axis('off')  # Hide axis

    # Show the colorbars with the original value range for both images
    fig.colorbar(im1, ax=axes[0], orientation='vertical', label='Original Values')
    fig.colorbar(im2, ax=axes[1], orientation='vertical', label='Original Values')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_two_contours(contour1, contour2, name1='Contour 1', name2='Contour 2'):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the first contour in blue
    ax.plot(contour1[:, 0], contour1[:, 1], label=name1, color='blue', marker='o')

    # Plot the second contour in red
    ax.plot(contour2[:, 0], contour2[:, 1], label=name2, color='red', marker='o')

    # Draw dotted lines connecting points with the same index
    for i in range(len(contour1)):
        ax.plot([contour1[i, 0], contour2[i, 0]], [contour1[i, 1], contour2[i, 1]], 'k:', lw=0.8)

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set the title and legend
    ax.set_title('Contours Plot')
    ax.legend()

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


def ff_get_shape_bckwrd_one_step(bin1, bin2, par):
    global ffX, ffY
    cellm = np.stack((bin1, bin2))
    pntsN = par.get('pntsN', 80)
    intType = par.get('intType', 'TPS')
    matchType = par.get('matchType', 'DTW')
    verbose = par.get('verbose', 0)
    meshtype = par.get('meshtype', 'inner')

    print(pntsN)

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

    cellm2 = cellm[0]

    cellB2 = get_contour_img(cellm2)
    P1 = GetContour(cellB2, pntsN)

    cellm2_2 = cellm[1]

    cellB2_2 = get_contour_img(cellm2_2)
    P2_not_sampled = GetContour(cellB2_2, -1)
    WP1, WP2 = MatchContours(P1, P2_not_sampled)

    # plot_two_contours(WP1, WP2, 'WP1', 'WP2')

    WP1_matlab = scipy.io.loadmat('.\\python_analog\\Series015_RA_WP1.mat')
    WP1_matlab = WP1_matlab['WP1']

    WP2_matlab = scipy.io.loadmat('.\\python_analog\\Series015_RA_WP2.mat')
    WP2_matlab = WP2_matlab['WP2']

    # print(WP1_matlab)

    plot_two_contours(WP1, WP1_matlab, 'WP1', 'WP1_matlab')
    plot_two_contours(WP2, WP2_matlab, 'WP2', 'WP2_matlab')

    ff, c = GetFlowField(WP2, WP1)
    # ff = -ff
    flowField.append([c, ff])

    if verbose <= 2:
        if useFEM == 2:
            ffX, ffY = algo_ff_built_FEM_LAME(ff, c, np.array(cellB2.shape), par.get('Young', 100000),
                                              0.4, par.get('triHmax', 15), meshtype)

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

image_path = '.\\python_analog\\Series015_RA_body.tif'
multi_layer_tiff = imageio.imread(image_path)
new_tiff = multi_layer_tiff.copy()
cellm = np.array(multi_layer_tiff, dtype=np.uint8)
cellm = cellm[:2:, :, :]
cellm[cellm == 0] = 255
cellm[cellm == 1] = 0
ffXics, ffYics, flowField = ff_get_shape_bckwrd_one_step(cellm[0], cellm[1], par)

plot_two_images(ffXics, ffYics)