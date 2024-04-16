import numpy as np
import matplotlib.pyplot as plt
from get_contour_discriptor import get_contour_descriptor
from find_shift import find_shift
import cv2

def match_contours(P1, P2, descTypeCorr, descTypeDTW, verbose=0, rotationLimit=np.pi / 2, alphaDTW=1,
                   resolveMultMatches=False, pntsScaleCorr=1):
    """
    Matches two contours P1 and P2. Matching is done in two stages.
    The first stage is rotation correction based on correlation of descTypeCorr descriptors.
    The second stage is Dynamic Time Warping based on descTypeDTW descriptors with multiple matches correction procedure.

    Parameters:
        P1: numpy array, first contour
        P2: numpy array, second\ contour
        descTypeCorr: str, descriptor type used for correlation rotation correction
        descTypeDTW: str, descriptor type used for Dynamic Time Warping, or 'none' if this stage is disabled
        verbose: int, show debug info: 0 - no info, 1 - only final contours, 2 - full info (default: 0)
        rotationLimit: float, value of possible maximum rotation for correlation correction in radians (default: pi/2)
        alphaDTW: float, "multiple to one" points penaltizer for DTW: alpha in [1,+inf) (default: 1)
        resolveMultMatches: bool, resolve 1-to-N matches by proportional resampling (default: False)
        pntsScaleCorr: int, the number tuning the number of points in descriptor used for correlation rotation compensation (default: 1)

    Returns:
        WP1: numpy array, first contour after matching
        WP2: numpy array, second contour after matching
        sh: int, shift size after correlation correction
    """
    if not descTypeCorr.lower() == 'none':
        # Convert rotationLimit from radians to shift size
        rotationLimit = int(rotationLimit * len(P1) / (2 * np.pi))

        d1corr = get_contour_descriptor(P1, descTypeCorr)
        d2corr = get_contour_descriptor(P2, descTypeCorr)

        sh = find_shift(d1corr, d2corr, rotationLimit)
        d2corr = np.roll(d2corr, -sh)

    else:
        sh = 0

    WP1 = P1.copy()
    WP2 = np.roll(P2.copy(), -sh, axis=0)

    # if verbose > 1:
    #     plotPs(WP1, WP2, 1, 1)

    # Resample back to pntsN number of points
    WP1 = WP1[::pntsScaleCorr]
    WP2 = WP2[::pntsScaleCorr]

    return WP1, WP2


# P1 = np.loadtxt('Contour1.dat')
# P2 = np.loadtxt('Contour2.dat')
# print(P1)
# descTypeCorr = 'Centroid'
# descTypeDTW = 'none'
# Verbose = 0
# rotationLimit = 125
# alphaDTW = 1.25
# resolveMultMatches = 1
# pntsScaleCorr = 1
#
# WP1, WP2, sh = match_contours(P1, P2, descTypeCorr, descTypeDTW, Verbose, rotationLimit, alphaDTW, resolveMultMatches, pntsScaleCorr)
#
#
# regParams = cv2.estimateAffinePartial2D(WP2.astype(np.float32), WP1.astype(np.float32))
#
# print(regParams[0])

