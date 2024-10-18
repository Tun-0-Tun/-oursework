import numpy as np
from scipy.interpolate import griddata
from getMesh import getMesh
from SolveFEM import solveFEM
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_triangulation(vertices, triangles, vertices2, triangles2):
    """
    Функция для отрисовки триангуляции.
    :param vertices: координаты вершин.
    :param triangles: индексы треугольников.
    """
    # Создание объекта триангуляции
    triangulation = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    triangulation2 = mtri.Triangulation(vertices2[:, 0], vertices2[:, 1], triangles2 - 1)

    # Построение триангуляции
    plt.figure(figsize=(8, 8))
    plt.triplot(triangulation, 'bo-', lw=1.5, color='red')
    plt.triplot(triangulation2, 'bo-', lw=1.5)
    plt.gca().set_aspect('equal')  # Установка равных пропорций осей
    plt.title("Триангуляция")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def algo_ff_built_FEM_LAME(ff, pboundary, sizeIm, E=100000, nu=0.03, triHmax=15, meshtype='inner'):
    if meshtype.lower() not in ['inner', 'outer', 'full']:
        raise ValueError("Invalid meshtype. It should be 'inner', 'outer', or 'full'.")

    if meshtype.lower() == 'inner':
        ptri, tri = getMesh(pboundary)
        # ff2 = scipy.io.loadmat('./Series015_RA_ff.mat')['ff']
        # ptri2 = scipy.io.loadmat('./Series015_RA_ptri.mat')['ptri']
        # tri2 = scipy.io.loadmat('./Series015_RA_tri.mat')['tri']
        # pboundary2 = scipy.io.loadmat('./Series015_RA_pboundary.mat')['pboundary']
        # fftri2, _ = solveFEM(pboundary2, ff, ptri2, tri2, E, nu)
        # plot_triangulation(ptri, tri,ptri2,tri2)
        # print(tri)
        # print(tri)
        fftri, _ = solveFEM(pboundary, ff, ptri, tri, E, nu)

    # fftri = fftri2
    # ptri = ptri2
    xx, yy = np.meshgrid(np.arange(1, sizeIm[1] + 1), np.arange(1, sizeIm[0] + 1))
    not_nan = ~np.isnan(fftri[:, 0])
    ffx = griddata((ptri[not_nan, 0], ptri[not_nan, 1]), fftri[not_nan, 0], (xx, yy), method='linear', fill_value=0)
    ffx[np.isnan(ffx)] = 0
    not_nan = ~np.isnan(fftri[:, 1])
    ffy = griddata((ptri[not_nan, 0], ptri[not_nan, 1]), fftri[not_nan, 1], (xx, yy), method='linear', fill_value=0)
    ffy[np.isnan(ffy)] = 0
    return ffx, ffy
