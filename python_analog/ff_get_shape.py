import imageio
import numpy as np
from matplotlib import pyplot as plt
from GetContour import get_contour_img, GetContour,get_contour_seq
from algo_ff_built_FEM_LAME import algo_ff_built_FEM_LAME
from scipy.spatial import cKDTree

import scipy.io
import os

from DefReg.utils.dots_remap_backward import algo_ff_remap_centroids_backward, remap_centroids_all_backward, remap_centroids_all_backward_0_k
from DefReg.utils.points_error_calculation import compute_l2_error_sequence


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
    cbar = fig.colorbar(im1, ax=axes[1], orientation='vertical', shrink=0.8, pad=0.05)
    cbar.set_label('Values')

    # Отобразить график
    plt.tight_layout()
    plt.show()


def plot_image_difference(image1, image2, t1='python', t2='MATLAB', cnt=1):
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
    axes[2].set_title(f'Difference {cnt}')
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
    verbose = par.get('verbose', 0)
    meshtype = par.get('meshtype', 'inner')


    if intType.upper() == 'FEM':
        useFEM = 2
    elif intType.upper() == 'TPS_GEO':
        useFEM = 1
    elif intType.upper() == 'TPS':
        useFEM = 0
    else:
        raise ValueError('Wrong interpolation type!')

    maxproj = np.sum(cellm, axis=0) > 0
    maxproj = (maxproj > 0).astype(int)
    m = measure(maxproj)[::2]
    ext_r = 5
    flowField = []

    cellB1 = get_contour_img(cellm1)
    cellB2 = get_contour_img(cellm2)

    P2 = GetContour(cellB2, pntsN)
    P1_not_sampled = GetContour(cellB1, 0)


    WP2, WP1 = MatchContours(P2, P1_not_sampled)


    ff, c = GetFlowField(WP2, WP1)
    flowField.append([c, ff])

    if verbose <= 2:
        if useFEM == 2:
            ffX, ffY = algo_ff_built_FEM_LAME(ff, c, np.array(cellB2.shape), par.get('Young', 100000),
                                              0.4, par.get('triHmax', 15), meshtype)
    return ffX, ffY, flowField


def ff_get_shape_bckwrd(cellm, par):
    base_cellm = cellm[0]

    ffXics = np.zeros_like(cellm).astype(float)
    ffYics = np.zeros_like(cellm).astype(float)
    seqLength = cellm.shape[0]
    print(seqLength)
    flowField = []

    for i in range(1, seqLength ):
        ffX_i, ffY_i, flowField_i = ff_get_shape_bckwrd_one_step(cellm[i - 1], cellm[i], par)
        ffXics[i - 1] = ffX_i
        ffYics[i - 1] = ffY_i
        flowField.append(flowField_i)
        print(f'Обработан {i} слой')
    return ffXics, ffYics




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
cellm = cellm
cellm[cellm == 0] = 255
cellm[cellm == 1] = 0
ffXics_python, ffYics_python = ff_get_shape_bckwrd(cellm, par)
'''
spots - это исходные unregistered точки
spotsRA - после жесткого преобразования
spotsReg - после жесткого+нежесткого из моей реализации матлабовской
'''
#=================Подготовка данных=============================
spotsI = scipy.io.loadmat('../../results to send/Series015/Series015_spotsI.mat')['spotsRA'][0]
spotsI = np.stack(spotsI)[:, :, :2]
spotsB = scipy.io.loadmat('../../results to send/Series015/Series015_spotsB.mat')['spotsRA'][0]
spotsB = np.stack(spotsB)[:, :, :2]
spotsL = scipy.io.loadmat('../../results to send/Series015/Series015_spotsL.mat')['spotsRA'][0]
spotsL = np.stack(spotsL)[:, :, :2]


# spotsL = np.stack(spotsL)[:, :, :2]
residualDistRegI = scipy.io.loadmat('../../results to send/Series015/Series015_spotsI.mat')['residualDistReg']
residualDistRegB = scipy.io.loadmat('../../results to send/Series015/Series015_spotsB.mat')['residualDistReg']
residualDistRegL = scipy.io.loadmat('../../results to send/Series015/Series015_spotsL.mat')['residualDistReg']
residualDistRegI = np.mean(residualDistRegI, axis=1)
residualDistRegB = np.mean(residualDistRegB, axis=1)
residualDistRegL = np.mean(residualDistRegL, axis=1)
#=================Подготовка данных=============================

#=================MATLAB fields================================
matlab_field_path = '../../results to send/Series015/Series015_ff_bcw.mat'
ffYics_matlab =  scipy.io.loadmat(matlab_field_path)['ffYics'][0][0][0]
ffXics_matlab = scipy.io.loadmat(matlab_field_path)['ffXics'][0][0][0]
field_MATLAB = np.stack((ffXics_matlab, ffYics_matlab), axis=-1)
field_MATLAB = np.transpose(field_MATLAB, (2, 0, 1, 3))
#=================MATLAB fields================================
# Создаём директорию data, если она не существует
os.makedirs('./data', exist_ok=True)
#=================Python fields================================
# ffXics_python, ffYics_python =  np.transpose(ffXics_python, (1, 2, 0)),  np.transpose(ffYics_python, (1, 2, 0))
# field_python = np.stack((ffXics_python, ffYics_python), axis=-1)
# field_python = np.transpose(field_python, (2, 0, 1, 3))
# field_python = field_python.astype(np.float32)
#=================Python fields================================
# np.save('./data/field_python.npy', field_python)
field_python = np.load('./data/field_python.npy')
print(field_python.shape)
print(field_MATLAB.shape)
#=================ДЕФОРМАЦИЯ================================
# поле деформаци в виде h, w, 2 -
reg_inner_MATLAB = remap_centroids_all_backward(spotsI, field_MATLAB)
reg_bound_MATLAB = remap_centroids_all_backward(spotsB, field_MATLAB)
reg_line_MATLAB = remap_centroids_all_backward(spotsL, field_MATLAB)

reg_inner_python = remap_centroids_all_backward(spotsI, field_python)
reg_bound_python= remap_centroids_all_backward(spotsB, field_python)
reg_line_python = remap_centroids_all_backward(spotsL, field_python)

#==================Подсчёт ошибок============================
base_inner_err_MATLAB = compute_l2_error_sequence(reg_inner_MATLAB)
base_bound_err_MATLAB = compute_l2_error_sequence(reg_bound_MATLAB)
base_line_err_MATLAB = compute_l2_error_sequence(reg_line_MATLAB)
base_inner_err_python = compute_l2_error_sequence(reg_inner_python)
base_bound_err_python = compute_l2_error_sequence(reg_bound_python)
base_line_err_python = compute_l2_error_sequence(reg_line_python)

np.savez('./data/errors_MATLAB.npz', inner=base_inner_err_MATLAB, bound=base_bound_err_MATLAB, line=base_line_err_MATLAB)
np.savez('./data/errors_Python.npz', inner=base_inner_err_python, bound=base_bound_err_python, line=base_line_err_python)


def plot_errors():
    indices = np.arange(len(base_inner_err_MATLAB))  # Индексы для x-оси

    plt.figure(figsize=(10, 6))

    # MATLAB ошибки
    plt.plot(indices, base_inner_err_MATLAB, label='MATLAB Inner Error', color='blue')
    plt.plot(indices, base_bound_err_MATLAB, label='MATLAB Bound Error', color='green')
    plt.plot(indices, base_line_err_MATLAB, label='MATLAB Line Error', color='red')

    # Python ошибки
    plt.plot(indices, base_inner_err_python, '--', label='Python Inner Error', color='blue')
    plt.plot(indices, base_bound_err_python, '--', label='Python Bound Error', color='green')
    plt.plot(indices, base_line_err_python, '--', label='Python Line Error', color='red')

    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.title('Errors Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('./data/error_plot.png')
    plt.show()


# plot_errors()

print("MATLAB")
print("bound",np.mean(base_bound_err_MATLAB))
print('inner', np.mean(base_inner_err_MATLAB))
print('line', np.mean(base_line_err_MATLAB))

print("Python")
print("bound", np.mean(base_bound_err_python))
print('inner',np.mean(base_inner_err_python))
print('line', np.mean(base_line_err_python))

res = get_contour_seq(cellm)

print(res.shape)
# imageio.mimwrite('cellm_contour.tiff', res, format='tiff')
# np.savez('cellm_contour.npz', res)



