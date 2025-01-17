import imageio
import numpy as np
from matplotlib import pyplot as plt
from GetContour import get_contour_img, GetContour
from algo_ff_built_FEM_LAME import algo_ff_built_FEM_LAME
from scipy.spatial import cKDTree
# from test_res import test_res


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


def ff_get_shape_bckwrd(cellm, par):
    pntsN = par.get('pntsN', 80)
    intType = par.get('intType', 'TPS')
    matchType = par.get('matchType', 'DTW')
    alphaDTW = par.get('alphaDTW', 1.25)
    resolveMultMatches = par.get('resolveMultMatches', 1)
    descTypeCorr = par.get('descTypeCorr', 'centroid')
    descTypeDTW = par.get('descTypeDTW', 'centroid')
    pntsScaleCorr = par.get('pntsScaleCorr', pntsN * 5)
    verbose = par.get('verbose', 0)
    debugPath = par.get('debugPath', '')
    ics = par.get('ics', [])
    meshtype = par.get('meshtype', 'inner')

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
    cellm = cellm[:, minY:maxY, minX:maxX]
    plt.imshow(cellm[0], cmap='gray')
    plt.show()

    ffXics = np.zeros_like(cellm).astype(float)
    ffYics = np.zeros_like(cellm).astype(float)
    seqLength = cellm.shape[0]
    flowField = []

    cellm2 = cellm[0]

    cellB2 = get_contour_img(cellm2)
    P2 = GetContour(cellB2, pntsN)
    for i in range(seqLength - 2):
        P1 = P2.copy()
        cellm2 = cellm[i + 1]

        cellB2 = get_contour_img(cellm2)
        P2_not_sampled = GetContour(cellB2, -1)

        WP1, WP2 = MatchContours(P1, P2_not_sampled)

        ff, c = GetFlowField(WP2, WP1)
        # ff = -ff
        flowField.append([c, ff])

        if verbose <= 2:
            if useFEM == 2:
                ffX, ffY = algo_ff_built_FEM_LAME(ff, c, np.array(cellB2.shape), par.get('Young', 100000),
                                                  0.4, par.get('triHmax', 15), meshtype)

        ffXics[i + 1] = ffX
        ffYics[i + 1] = ffY

    return ffXics, ffYics, flowField


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
image_path = '.\\python_analog\Series015_RA_body.tif'
multi_layer_tiff = imageio.imread(image_path)
new_tiff = multi_layer_tiff.copy()
cellm = np.array(multi_layer_tiff, dtype=np.uint8)
cellm[cellm == 0] = 255
cellm[cellm == 1] = 0
ffXics, ffYics, flowField = ff_get_shape_bckwrd(cellm, par)

np.save('ffXics_result.npy', ffXics)
np.save('ffYics_result.npy', ffYics)
# test_res()