import imageio
import numpy as np
from matplotlib import pyplot as plt
from GetContour import get_contour_img, GetContour
from algo_ff_built_FEM_LAME import algo_ff_built_FEM_LAME
from scipy.spatial import cKDTree
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


    # Add more properties as needed

    return minimum, maximum

def MatchContours(contour1, contour2):
    tree = cKDTree(contour2)
    distances, indices = tree.query(contour1)
    nearest_points = contour2[indices]
    return contour1, nearest_points
# def MatchContours(p1, p2_not_sampled):
#     wp2 = np.zeros_like(p1)
#     for i in range(p1.shape[1]):
#         wp2[:, i] = find_nearest_points(p2_not_sampled, p1[:, i])
#     wp1 = p1
#     return wp1, wp2


def ff_get_shape_bckwrd(cellm, par):
    # Constructs the backward deformation field based using Shape Registration
    # Method (DeVylder or my)
    #
    # synopsys:
    # [ffXics ffYics flowField] = ff_get_shape_bckwrd(cellm, par)
    #
    # outputs:
    # ffXics -      interpolated ics stack of deformation fields (x component)
    # ffYics -      interpolated ics stack of deformation fields (y component)
    # flowField -   shape deformation fields for the whole stack
    #
    # inputs:
    # cellm -       stack of body images
    # par -         parameters structure
    #
    # parameters structure can have the folowwing fields:
    #
    # pntsN -       number of points to sample the boundary of the cell
    # intType -            The type of deformation field interpolation to use
    #                      'FEM' or 'TPS'
    # Young -              Young modulus for FEM interpolation (default 100000)
    # Poisson -            Poisson ratio for FEM interpolation (default 0.5)
    # triHmax -            maximum size of triangulation element for FEM
    #                      interpolation (default 15)
    # matchType -          contour matching type: 'DTW' or 'DistTransform'
    # alphaDTW -           alpha parameter for DTW method (must be >1, 1.25 is
    #                      mostly reccommended)
    # resolveMultMatches - 1|0. Parameter to control resolving multiple matches
    #                      in contour points matching
    # descTypeCorr -       shape descriptor type for correcting shift using
    #                      correlation ('none', 'complex', 'centroid',
    #                      'tangentdirection', 'curvature')
    # descTypeDTW -        shape descriptor type for correcting shift using
    #                      DTW ('none', 'complex', 'centroid',
    #                      'tangentdirection', 'curvature')
    # pntsScaleCorr -      The number tuning the number of points in descriptor
    #                      used for correlation rotation compensation
    #                      (pntsNcorr = pntsN*pntsScaleCorr)
    # ics -                 intensity image stack. if not empty - contour enhancement
    #                       option is activated
    # meshtype -           'full' | 'inner' | 'outer'. if to compute
    #                      deformation fields inside the cell, outside or
    #                      everywhere
    # verbose -            1|0 - on|off, 2 - debug mode (show contours and def.
    #                      field), 3 - debug mode (show just matched contours)
    #                      and full process of matching, 4 - show just matched
    #                      contours

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

    cellmInit = cellm.copy()

    # crop the image not to process the rest of the data
    maxproj = np.sum(cellm, axis=0) > 0
    maxproj = (maxproj > 0).astype(int)
    # plt.imshow(maxproj, cmap='gray')
    # plt.show()
    m = measure(maxproj)[::2]
    ext_r = 5
    minX = max(0, m[0][1] - ext_r)
    minY = max(0, m[0][0] - ext_r)
    maxX = min(cellm.shape[2], m[1][1] + ext_r)
    maxY = min(cellm.shape[1], m[1][0] + ext_r)
    cellm = cellm[:, minY:maxY, minX:maxX]
    plt.imshow(cellm[0], cmap='gray')
    plt.show()

    ffXics = np.zeros_like(cellm)
    ffYics = np.zeros_like(cellm)
    seqLength = cellm.shape[0]
    flowField = []

    cellm2 = cellm[0]

    cellB2 = get_contour_img(cellm2)
    P2 = GetContour(cellB2, pntsN)
    print(f'Computing FF. Overall {seqLength-2} frames.  Progress: ')
    for i in range(seqLength-2):
        cellm1 = cellm2.copy()
        cellB1 = cellB2.copy()
        P1 = P2.copy()
        cellm2 = cellm[i + 1]

        cellB2 = get_contour_img(cellm2)
        P2_not_sampled = GetContour(cellB2, -1)

        #vv = max([verbose == 4, (verbose == 3) + 1, (verbose == 2) + 1, (verbose == 1) + 1]) * (verbose != 0)
        # if useDT:
        #     WP1, _ = MatchContoursDT(P2, cellm1, cellm2, cellB1, cellB2, vv)
        #     WP2 = P2
        #     if ics:
        #         WP2, WP1 = enhanceContours(WP2, WP1, gaussf(cellI2, 3), gaussf(cellI1, 3), 0)
        # else:
        #     WP1, WP2 = MatchContours(P1, P2, descTypeCorr, descTypeDTW,
        #                              vv, np.pi / 2, alphaDTW, resolveMultMatches, pntsScaleCorr)
        WP1, WP2 = MatchContours(P1, P2_not_sampled)


        ff, c = GetFlowField(WP2, WP1)
        flowField.append([c, ff])

        if verbose <= 2:
            if useFEM == 2:
                ffX, ffY = algo_ff_built_FEM_LAME(ff, c, np.array(cellB2.shape), par.get('Young', 100000),
                                                  0.4, par.get('triHmax', 15), meshtype)


        print(ffX, ffY)

        ffXics[i + 1,: ,:] = ffX
        ffYics[i + 1,:, :] = ffY

        # if verbose > 0:
        #     printdbg = not debugPath or verbose == 4 or verbose == 2
        #     if not printdbg:
        #         plt.show()
        #     else:
        #         plt.savefig(f'{debugPath}/{i:04d}.png')
        #         np.savez(f'{debugPath}/ptns/{i:04d}.npz', P1=P1, P2=P2, WP1=WP1, WP2=WP2, descTypeCorr=descTypeCorr,
        #                  descTypeDTW=descTypeDTW, alphaDTW=alphaDTW)
        #         plt.close()
    print('\n')

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
    'ics': None,  # Предполагаем, что нет изображений интенсивности
    'meshtype': 'inner'
}
image_path = 'C:\\Users\\alexp\\Артём\\Курсовая\\Coursework\\python_analog\\dataset\\Series015_RA_body.tif'
multi_layer_tiff = imageio.imread(image_path)
new_tiff = multi_layer_tiff.copy()
cellm = np.array(multi_layer_tiff, dtype=np.uint8)
cellm[cellm == 0] = 255
cellm[cellm == 1] = 0
# plt.imshow(cellm[1], cmap='gray')
# plt.show()
ffXics, ffYics, flowField = ff_get_shape_bckwrd(cellm,  par)

print(ffXics, ffYics)