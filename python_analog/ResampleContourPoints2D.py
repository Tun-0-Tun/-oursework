import numpy as np
from scipy.interpolate import interp1d


def ResampleContourPoints2D(P, nPoints):
    # Interpolate points inbetween
    O = np.vstack([P, P[0]])
    dis = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(np.diff(O, axis=0) ** 2, axis=1)))])

    # Resample to make uniform points
    interp_x = interp1d(dis, O[:, 0], kind='linear')
    interp_y = interp1d(dis, O[:, 1], kind='linear')

    K = np.column_stack([interp_x(np.linspace(0, dis[-1], nPoints + 1)),
                         interp_y(np.linspace(0, dis[-1], nPoints + 1))])

    return K[:-1]
