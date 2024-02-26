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
    sforce[np.where(n[1] == 1)] = 1
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


# Initialisation
E = 2900
nu = 0.4
mu = E / (2 * (1 + nu))
lambda_val = E * nu / ((1 + nu) * (1 - 2 * nu))
coordinates = np.loadtxt('coordinates.dat')
elements3 = np.loadtxt('elements3.dat', ndmin=2, unpack=True)
elements4 = np.loadtxt('elements4.dat', ndmin=2, unpack=True)
neumann = np.loadtxt('neumann.dat', ndmin=2, unpack=True)
dirichlet = np.loadtxt('dirichlet.dat')
A = np.zeros((2 * len(coordinates), 2 * len(coordinates)))
b = np.zeros(2 * len(coordinates))

# Assembly
for j in range(elements3.shape[0]):
    I = 2 * elements3[j, [0, 0, 1, 1, 2, 2]] - np.array([1, 0, 1, 0, 1, 0])
    A[I[:, None], I] += stima3(coordinates[elements3[j]], lambda_val, mu)

for j in range(elements4.shape[0]):
    I = 2 * elements4[j, [0, 0, 1, 1, 2, 2, 3, 3]] - np.array([1, 0, 1, 0, 1, 0, 1, 0])
    A[I[:, None], I] += stima4(coordinates[elements4[j]], lambda_val, mu)

# Volume forces
for j in range(elements3.shape[0]):
    I = 2 * elements3[j, [0, 0, 1, 1, 2, 2]] - np.array([1, 0, 1, 0, 1, 0])
    fs = f(np.sum(coordinates[elements3[j]], axis=0) / 3)
    b[I] += np.dot(np.array([1, 1, 1]), coordinates[elements3[j]].T) * np.array([fs, fs, fs]) / 6

for j in range(elements4.shape[0]):
    I = 2 * elements4[j, [0, 0, 1, 1, 2, 2, 3, 3]] - np.array([1, 0, 1, 0, 1, 0, 1, 0])
    fs = f(np.sum(coordinates[elements4[j]], axis=0) / 4)
    b[I] += np.dot(np.array([1, 1, 1]), coordinates[elements4[j, :3]].T) * np.array([fs, fs, fs, fs]) / 4

# Neumann conditions
if neumann.size > 0:
    n = (coordinates[neumann[:, 1]] - coordinates[neumann[:, 0]]) @ np.array([[0, -1], [1, 0]])
    for j in range(neumann.shape[0]):
        I = 2 * neumann[j, [0, 0, 1, 1]] - np.array([1, 0, 1, 0])
        gm = g(np.sum(coordinates[neumann[j]], axis=0) / 2, n[j] / np.linalg.norm(n[j]))
        b[I] += np.linalg.norm(n[j]) * np.array([gm, gm]) / 2

# Dirichlet conditions
DirichletNodes = np.unique(dirichlet)
W, M = u_d(coordinates[DirichletNodes])
B = np.zeros((W.shape[0], 2 * len(coordinates)))
for k in range(2):
    for l in range(2):
        B[1 + l::2, 2 * DirichletNodes - 1 + k] = np.diag(M[1 + l::2, k])



# mask = np.where(np.sum(np.abs(B), axis=1))[0]
# A = np.block([[A, B[mask].T], [B[mask], np.zeros((len(mask), len(mask))]])
# b = np.concatenate((b, W[mask]))
#
# # Calculating the solution
# x = np.linalg.solve(A, b)
# u = x[:2 * len(coordinates)]
#
# # Representation of the solution
# AvE, Eps3, Eps4, AvS, Sigma3, Sigma4 = avmatrix(coordinates, elements3, elements4, u, lambda_val, mu)
# show(elements3, elements4, coordinates, AvS, u, lambda_val, mu)
# estimate = aposteriori(coordinates, elements3, elements4, AvE, Eps3, Eps4, AvS, Sigma3, Sigma4, u, lambda_val, mu)

