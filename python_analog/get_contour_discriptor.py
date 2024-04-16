import numpy as np


def get_contour_descriptor(P, descType='centroid'):
    """
    Gets shape descriptor from given contour

    Parameters:
        P: numpy array, shape (n, 2), contour points
        descType: str, type of descriptor: 'complex', 'centroid', 'tangentdirection', or 'curvature'

    Returns:
        desc: numpy array, shape depends on descriptor type
    """
    x = P[:, 0]
    y = P[:, 1]
    xc = np.sum(x) / len(x)
    yc = np.sum(y) / len(y)

    if descType.lower() == 'complex':
        desc = x + 1j * y
    elif descType.lower() == 'centroid':
        desc = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) / np.sum(np.sqrt((x - xc) ** 2 + (y - yc) ** 2)) * len(x)
    elif descType.lower() == 'tangentdirection':
        x = np.concatenate(([x[-1]], x, [x[0]]))
        y = np.concatenate(([y[-1]], y, [y[0]]))
        desc = np.arccos((y[2:] - y[:-2]) / np.sqrt((x[2:] - x[:-2]) ** 2 + (y[2:] - y[:-2]) ** 2))
    elif descType.lower() == 'curvature':
        diffx = np.diff(np.concatenate(([x[-1]], x)))
        diffy = np.diff(np.concatenate(([y[-1]], y)))
        diff2x = np.diff(np.concatenate(([diffx[-1]], diffx)))
        diff2y = np.diff(np.concatenate(([diffy[-1]], diffy)))
        numerator = diffx * diff2y - diff2x * diffy
        denominator = (diffx ** 2 + diffy ** 2) * np.sqrt(diffx ** 2 + diffy ** 2)
        desc = numerator / denominator
    else:
        raise ValueError(f"Unknown descriptor type!\n descType = {descType}")

    return desc


# test
coordinates = np.loadtxt('test_disk.dat')
print(get_contour_descriptor(coordinates))
