import numpy as np
import matplotlib.pyplot as plt

def make_proportional_resampling_new(P1, P2, verbose=False):
    """
    Resamples the points of P2 curve which have equal values with knots
    proportional according to the distance between corresponding P1 points

    Parameters:
        P1: numpy array, curve that used for taking the sampling distances
        P2: numpy array, curve that has equal values needed to be resampled
        verbose: bool, whether to display additional information for debugging (default: False)

    Returns:
        P2res: numpy array, result with equal values interpolated between the prev and next values
    """
    P1sizeFlag = (P1.shape[1] == 1)
    P2sizeFlag = (P2.shape[1] == 1)

    if P1sizeFlag and P2sizeFlag:
        P1 = np.column_stack((P1, P1))
        P2 = np.column_stack((P2, P2))
    if sum(np.logical_xor(P1sizeFlag, P2sizeFlag)):
        raise ValueError('Array sizes must be equal!')

    # Find indexes of repeated elements in P2ind (find which points correspond to multiple)
    _, _, repInd = np.unique(P2, axis=0, return_inverse=True)
    n, binEdges = np.histogram(repInd, bins=np.arange(np.max(repInd) + 2))
    binInd = np.digitize(repInd, binEdges) - 1
    multiple = np.where(n > 1)[0]

    while len(multiple) > 0:
        ind = np.where(binInd == multiple[0])[0]

        # if verbose:
        #     plotPs(P1, P2, 1, 1)

        maxInd = P2.shape[0]
        indWithNeigh = np.concatenate(([ind[0] - 1], ind, [ind[-1] + 1]))
        indWithNeigh[indWithNeigh > maxInd - 1] -= maxInd
        indWithNeigh[indWithNeigh < 0] += maxInd

        P1part = P1[indWithNeigh]
        P2part = P2[indWithNeigh]
        P2partNew = np.zeros_like(P2part)

        if verbose:
            plt.plot(P2part[:, 0], P2part[:, 1], '.r')
            plt.plot(P1part[:, 0], P1part[:, 1], '.c')

        cumdistP1 = np.concatenate(([0], np.cumsum(np.sqrt(np.sum((P1part[1:] - P1part[:-1]) ** 2, axis=1)))))
        cumdistP2 = np.concatenate(([0], np.cumsum(np.sqrt(np.sum((P2part[1:] - P2part[:-1]) ** 2, axis=1)))))
        sumdistP1 = cumdistP1[-1]
        sumdistP2 = cumdistP2[-1]

        indP2_j = cumdistP1 / sumdistP1 < cumdistP2[1] / sumdistP2
        vect = (P2part[1] - P2part[0]) / np.linalg.norm(P2part[1] - P2part[0]) * sumdistP2
        P2partNew[indP2_j] = np.tile(P2part[0], (np.sum(indP2_j), 1)) + np.tile(vect, (np.sum(indP2_j), 1)) * (
                    cumdistP1[indP2_j] / sumdistP1)

        indP2_j = ~indP2_j
        vect = (P2part[1] - P2part[-1]) / np.linalg.norm(P2part[1] - P2part[-1]) * sumdistP2
        P2partNew[indP2_j] = np.tile(P2part[-1], (np.sum(indP2_j), 1)) + np.tile(vect, (np.sum(indP2_j), 1)) * (
        (1 - cumdistP1[indP2_j] / sumdistP1))

        P2[indWithNeigh] = P2partNew

        if verbose:
            plt.plot(P2partNew[:, 0], P2partNew[:, 1], 'om')

        _, _, repInd = np.unique(P2, axis=0, return_inverse=True)
        n, binEdges = np.histogram(repInd, bins=np.arange(np.max(repInd) + 2))
        binInd = np.digitize(repInd, binEdges) - 1
        multiple = np.where(n > 1)[0]

    P2res = P2
    if P2sizeFlag:
        P2res = P2res[:, [0]]

    return P2res
