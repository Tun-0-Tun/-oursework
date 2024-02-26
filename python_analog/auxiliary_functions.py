import numpy as np


def stima3(vertices, lmbda, mu):
    # Compute the gradients of the basis functions
    PhiGrad = \
    np.linalg.lstsq(np.vstack((np.ones((1, 3)), vertices.T)), np.vstack((np.zeros((1, 2)), np.eye(2))), rcond=None)[0]

    # print(PhiGrad)
    # print(np.vstack((np.ones((1, 3)), vertices.T)))
    # print(np.vstack((np.zeros((1, 2)), np.eye(2))))


    # Construct the strain-displacement matrix
    R = np.zeros((3, 6))
    R[np.ix_([0, 2], [0, 2, 4])] = PhiGrad.T
    R[np.ix_([2, 1], [1, 3, 5])] = PhiGrad.T

    # Construct the material stiffness matrix
    C = mu * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) + lmbda * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])

    # Compute the element stiffness matrix
    stima3 = np.linalg.det(np.vstack((np.ones(3), vertices.T))) / 2 * np.dot(R.T, np.dot(C, R))

    return stima3


def stima4(vertices, lambda_val, mu_val):
    R_11 = np.array([[2, -2, -1, 1],
                     [-2, 2, 1, -1],
                     [-1, 1, 2, -2],
                     [1, -1, -2, 2]]) / 6

    R_12 = np.array([[1, 1, -1, -1],
                     [-1, -1, 1, 1],
                     [-1, -1, 1, 1],
                     [1, 1, -1, -1]]) / 4

    R_22 = np.array([[2, 1, -1, -2],
                     [1, 2, -2, -1],
                     [-1, -2, 2, 1],
                     [-2, -1, 1, 2]]) / 6

    F = np.linalg.inv(np.vstack((vertices[1] - vertices[0], vertices[3] - vertices[0])))
    L = np.array([lambda_val + 2 * mu_val, lambda_val, mu_val])
    stima4 = np.zeros((8, 8))

    E = np.dot(np.dot(F.T, np.array([[L[0], 0], [0, L[2]]])), F)
    stima4[0:7:2, 0:7:2] = E[0, 0] * R_11 + E[0, 1] * R_12 + E[1, 0] * R_12.T + E[1, 1] * R_22

    E = np.dot(np.dot(F.T, np.array([[L[2], 0], [0, L[0]]])), F)
    stima4[1:8:2, 1:8:2] = E[0, 0] * R_11 + E[0, 1] * R_12 + E[1, 0] * R_12.T + E[1, 1] * R_22

    E = np.dot(np.dot(F.T, np.array([[0, L[2]], [L[1], 0]])), F)
    stima4[1:8:2, 0:7:2] = E[0, 0] * R_11 + E[0, 1] * R_12 + E[1, 0] * R_12.T + E[1, 1] * R_22

    stima4[0:7:2, 1:8:2] = stima4[1:8:2, 0:7:2].T
    stima4 = stima4 / np.linalg.det(F)

    return stima4


def f(x):
    volforce = np.zeros_like(x)
    return volforce


def g(x, n):
    sforce = np.zeros_like(x)
    sforce[np.nonzero(n[1] == 1)] = 1
    return sforce


def u_d(x):
    N = x.shape[0]
    M = np.zeros((2 * N, 2))
    W = np.zeros((2 * N, 1))

    # symmetry conditions on the x-axis
    temp = np.where((x[:, 0] > 0) & (x[:, 1] == 0))[0]
    M[2 * temp - 1, 1] = 1

    # symmetry conditions on the y-axis
    temp = np.where((x[:, 1] > 0) & (x[:, 0] == 0))[0]
    M[2 * temp - 1, 0] = 1

    return W, M
