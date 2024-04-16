import numpy as np
from skimage.io import imread
from scipy.spatial import distance


# Функция для чтения изображений в формате TIFF
def read_tiff(filename):
    return imread(filename)


# Функция для измерения периметра и размеров эллипсоида
def measure_image_properties(image):
    perimeter = np.sum(image)
    dimensions_ellipsoid = np.array(image.shape)
    size = np.sum(image != 0)
    return perimeter, dimensions_ellipsoid, size


# Функция для регистрации последовательности изображений
def register_sequence(images, parameters, filename, result_path):
    # Здесь должна быть реализация вашего алгоритма регистрации

    # Пример вывода результатов
    print("Registered sequence with parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")

    print("Filename:", filename)
    print("Result path:", result_path)


# Имя файла
fname = './dataset/Series015.tif'
res_path_prefix = 'pntsN = 100_journal_E=1e4_P=0.4'

# Параметры материала
young = 1e4
poiss = 0.4

# Суффикс файла с изображениями
cellm_suffix = '_body.tif'

# Чтение изображений
cellm = read_tiff(fname[:-4] + cellm_suffix)

# Параметры
parameters = {
    'verbose': 0,
    'NonRigid': 1,
    'recomputeFF': 1,
    'recomputeSpots': 1,
    'recomputeRigid': 1,
    'debugPath': '',
    'resPathPref': res_path_prefix,
    'startInd': 0,
    'finishInd': cellm.shape[2] - 1,
    'seqLength': cellm.shape[2],
    'RAnum': 2,
    'RAtype': 'AO',
    'matchType': 'DTW',
    'intType': 'TPS',
    'Young': young,
    'Poisson': poiss,
    'triHmax': 15,
    'pntsN': 100,
    'pntsScaleCorr': 1,
    'alphaDTW': 1.25,
    'resolveMultMatches': 1,
    'descTypeCorr': 'centroid',
    'descTypeDTW': 'centroid'
}

# Измерение свойств изображения
perimeter, dimensions_ellipsoid, size = measure_image_properties(cellm[:, :, 0])
cell_stat = {
    'contour_sampling_dist': perimeter / parameters['pntsN'],
    'cell_ellipsoid_size': dimensions_ellipsoid,
    'cell_area': size
}

# Добавление статистики в параметры
parameters['cellStat'] = cell_stat

# Обрезка изображений
cellm = cellm[:, :, parameters['startInd']:parameters['finishInd'] + 1]

# Регистрация последовательности
# register_sequence(cellm, parameters, fname, "RA - DistT - FEM")
print("----")
