import numpy as np
from scipy.sparse import csr_matrix
from auxiliary_functions import stima3, stima4, f, g
from scipy.sparse import vstack, hstack


def u_d(ff):
    """
    Data on the Dirichlet boundary.

    Parameters:
        ff (numpy.ndarray): Displacement field on the Dirichlet boundary.
                            It has dimension N x 2.

    Returns:
        W (numpy.ndarray): Direction for which the displacement is given.
                            It has dimension 2*N x 1.
        M (numpy.ndarray): Corresponding values. If U is the displacement vector,
                            the Dirichlet boundary condition is given by M*U = W.
                            It has dimension 2*N x 2.
    """
    N = ff.shape[0]
    M = np.zeros((2 * N, 2))
    W = np.zeros((2 * N, 1))

    M[::2, 0] = 1
    M[1::2, 1] = 1

    W[::2, 0] = ff[:, 0]
    W[1::2, 0] = ff[:, 1]

    return W, M


def solveFEM(Pb, ff, P, el3, E, nu):
    """
    Solves linear Navier equation.

    Parameters:
        Pb (numpy.ndarray): Boundary points
        ff (numpy.ndarray): Deformation field in the boundary points
        P (numpy.ndarray): Triangulation points
        el3 (numpy.ndarray): Triangles (finite elements)
        E (float): Young's modulus
        nu (float): Poisson ratio

    Returns:
        U (numpy.ndarray): Deformation field in the inner points of the region
        Ub (numpy.ndarray): Deformation field on the boundary
    """

    mu = E / (2 * (1 + nu))
    lambda_val = E * nu / ((1 + nu) * (1 - 2 * nu))

    node_boundary = np.where(np.in1d(P[:, 0], Pb[:, 0]) & np.in1d(P[:, 1], Pb[:, 1]))[0]
    iii = []
    for n_b in node_boundary:
        cur_b = P[n_b]
        index = np.where((Pb == cur_b).all(axis=1))[0]
        iii.append(index[0])

    ff2 = ff[iii]
    el3 = el3[:, :3]

    A = csr_matrix((2 * P.shape[0], 2 * P.shape[0]))
    b = np.zeros(2 * P.shape[0])

    # Assembly
    for j in range(el3.shape[0]):
        I = 2 * el3[j, [0, 0, 1, 1, 2, 2]] - np.array([1, 0, 1, 0, 1, 0])
        A[I[:, np.newaxis] + 1, I + 1] += stima3(P[el3[j]], lambda_val, mu)

    for j in range(el3.shape[0]):
        I = 2 * el3[j, [0, 0, 1, 1, 2, 2]] - [1, 0, 1, 0, 1, 0]
        fs = np.array([0, 0])
        b[I] += np.linalg.det(np.hstack((np.ones((3, 1)), P[el3[j]]))) * np.tile(fs, 3) / 6

    DirichletNodes = node_boundary
    W, M = u_d(ff2)
    B = np.zeros((W.shape[0], 2 * P.shape[0]))

    M_rows, M_cols = M.shape

    for k in range(2):
        for l in range(2):
            diag_values = M[l:M_rows:2, k]
            B[l:M_rows:2, 2 * DirichletNodes - 1 + k] = np.diag(diag_values.flatten())
    B = np.roll(B, 1, axis=1)
    mask = np.where(np.sum(np.abs(B), axis=1))[0]

    A_top = hstack([A, B[mask].T])
    A_bottom = hstack([B[mask], csr_matrix((len(mask), len(mask)), dtype=np.float64)])
    A = vstack([A_top, A_bottom])
    b = b.reshape(-1, 1)
    b_masked = W[mask]
    b = np.vstack([b, b_masked])

    x = np.linalg.lstsq(A.toarray(), b, rcond=None)[0]
    u = x[:2 * len(P)]

    Ub = np.column_stack((W[::2], W[1::2]))  # Выбираем четные и нечетные элементы массива W
    U = np.column_stack((u[::2], u[1::2]))  # Выбираем четные и нечетные элементы массива u

    return U, Ub

# Pb = np.loadtxt('./FEM_data/Pb.dat', dtype=float)
# ff = np.loadtxt('./FEM_data/ff.dat', dtype=float)
# P = np.loadtxt('./FEM_data/P.dat', dtype=float)
# el3 = np.loadtxt('./FEM_data/el3.dat', dtype=int) - 1
# E = 10000
# nu = 0.4
# solveFEM(Pb, ff, P, el3, E, nu)
