import numpy as np
import scipy.signal as signal


def find_shift(c1, c2, r):
    """
    Finds shift between 2 circular periodic curves based on correlation
    in assumption that it can't be more than r

    Parameters:
        c1: numpy array, first curve
        c2: numpy array, second curve
        r: int, maximum possible shift: abs(sh) <= r

    Returns:
        sh: int, shift between curves
    """
    if c1.ndim < 2:
        c1 = c1.reshape(-1, 1)
    if c2.ndim < 2:
        c2 = c2.reshape(-1, 1)
    if c1.shape[1] > 1 or c2.shape[1] > 1:
        raise ValueError('Input parameter should be vectors!')
    if len(c1) < len(c2):
        raise ValueError('Length of c1 must be greater or equal of length of c2!')

    c1rep = np.tile(c1, (2, 1))
    N1 = len(c2)
    N2 = len(c1rep)
    c2_padded = np.pad(c2.flatten(), (0, len(c1rep.flatten()) - len(c2.flatten())), mode='constant')
    corr = signal.correlate(c1rep.flatten(), c2_padded, mode='full')
    # corr = signal.correlate(c1rep.flatten(),  c2.flatten(), mode="full")
    corr1 = corr[N2 - 1:N2 + N1 - 1] + 1
    corr1[r :-r] = 0
    sh = N1 - np.argmax(corr1)
    sh %= N1
    if sh > N1 / 2:
        sh -= N1

    return sh


p1 = np.loadtxt('p1.dat')
p2 = np.loadtxt('p2.dat')
print(find_shift(p1, p2, 125))
