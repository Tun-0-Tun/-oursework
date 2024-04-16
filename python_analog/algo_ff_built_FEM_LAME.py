import numpy as np
from scipy.interpolate import griddata
from getMesh import getMesh
from SolveFEM import solveFEM

def algo_ff_built_FEM_LAME(ff, pboundary, sizeIm, E=100000, nu=0.03, triHmax=15, meshtype='inner'):
    if meshtype.lower() not in ['inner', 'outer', 'full']:
        raise ValueError("Invalid meshtype. It should be 'inner', 'outer', or 'full'.")

    if meshtype.lower() == 'inner':
        ptri, tri = getMesh(pboundary)
        fftri, _ = solveFEM(pboundary, ff, ptri, tri, E, nu)
    xx, yy = np.meshgrid(np.arange(1, sizeIm[1] + 1), np.arange(1, sizeIm[0] + 1))
    not_nan = ~np.isnan(fftri[:, 0])
    ffx = griddata((ptri[not_nan, 0], ptri[not_nan, 1]), fftri[not_nan, 0], (xx, yy), method='linear', fill_value=0)
    ffx[np.isnan(ffx)] = 0
    not_nan = ~np.isnan(fftri[:, 1])
    ffy = griddata((ptri[not_nan, 0], ptri[not_nan, 1]), fftri[not_nan, 1], (xx, yy), method='linear', fill_value=0)
    ffy[np.isnan(ffy)] = 0
    return ffx, ffy
