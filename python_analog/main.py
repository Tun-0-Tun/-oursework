import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, hstack

from auxiliary_functions import stima3, stima4, f, g, u_d

E = 2900
nu = 0.4
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

print("mu", mu)
print("lmbda", lmbda)

coordinates = np.loadtxt('coordinates.dat')
elements3 = np.loadtxt('elements3.dat', dtype=int) - 1
elements4 = np.loadtxt('elements4.dat', dtype=int) - 1
neumann = np.loadtxt('neumann.dat', dtype=int) - 1
dirichlet = np.loadtxt('dirichlet.dat', dtype=int) - 1

n_coords = len(coordinates)
print('Число координат:', n_coords)
A = csr_matrix((2 * n_coords, 2 * n_coords), dtype=float)
b = np.zeros((2 * n_coords, 1))

A_lil = A.tolil()

# Assembly
for j in range(elements3.shape[0]):
    I = 2 * elements3[j, [0, 0, 1, 1, 2, 2]] - np.array([1, 0, 1, 0, 1, 0])
    A_lil[I[:, np.newaxis] + 1, I + 1] += stima3(coordinates[elements3[j]], lmbda, mu)


for j in range(elements4.shape[0]):
    I = 2 * elements4[j, [0, 0, 1, 1, 2, 2, 3, 3]]- np.array([1, 0, 1, 0, 1, 0, 1, 0])
    # print(stima4(coordinates[elements4[j]], lmbda, mu))
    A_lil[I[:, np.newaxis] + 1, I + 1] += stima4(coordinates[elements4[j]], lmbda, mu)


A = A_lil.tocsr()
# print(A_lil)

for j in range(elements3.shape[0]):
    I = 2 * elements3[j, [0, 0, 1, 1, 2, 2]] - np.array([1, 0, 1, 0, 1, 0])
    fs = f(np.sum(coordinates[elements3[j]], axis=0) / 3)
    fs_rep = np.tile(fs[:, np.newaxis], (3, 1))  # Повторяем вектор fs, чтобы он имел размерность (6, 1)
    b[I] += np.linalg.det(np.hstack((np.ones((3, 1)), coordinates[elements3[j]]))) * fs_rep / 6

for j in range(elements4.shape[0]):
    I = 2 * elements4[j, [0, 0, 1, 1, 2, 2, 3, 3]] - np.array([1, 0, 1, 0, 1, 0, 1, 0])
    fs = f(np.sum(coordinates[elements4[j, :3]], axis=0) / 4)
    fs_rep = np.tile(fs[:, np.newaxis], (4, 1))  # Повторяем вектор fs, чтобы он имел размерность (4, 1)
    b[I] += np.linalg.det(np.hstack((np.ones((3, 1)), coordinates[elements4[j, :3]]))) * fs_rep / 4



if neumann.size != 0:
    n = (coordinates[neumann[:, 1]] - coordinates[neumann[:, 0]]) @ np.array([[0, -1], [1, 0]])
    for j in range(neumann.shape[0]):
        I = 2 * neumann[j, [0, 0, 1, 1]] - np.array([1, 0, 1, 0])
        gm = g(np.sum(coordinates[neumann[j]], axis=0) / 2, n[j] / np.linalg.norm(n[j])).flatten()
        b[I] += np.linalg.norm(n[j]) * np.array([gm, gm]).flatten().reshape(-1, 1) / 2


DirichletNodes = np.unique(dirichlet)
W, M = u_d(coordinates[DirichletNodes, :])

B = np.zeros((M.shape[0], 2 * coordinates.shape[0] ))

M_rows, M_cols = M.shape

M = np.roll(M, 1, axis=0)

for k in range(2):
    for l in range(2):
        diag_values = M[l:M_rows:2, k ]
        B[l:M_rows:2, 2 * DirichletNodes - 1 + k] = np.diag(diag_values.flatten())

B = np.roll(B, 1)

mask = np.where(np.sum(np.abs(B), axis=1))[0]

A_top = hstack([A, B[mask].T])
A_bottom = hstack([B[mask], csr_matrix((len(mask), len(mask)), dtype=np.float64)])
A = vstack([A_top, A_bottom])

b_masked = W[mask]
b = np.vstack([b, b_masked])
b = np.roll(b, 2)
# print(csr_matrix(b))

x = np.linalg.lstsq(A.toarray(), b, rcond=None)[0]
u = x[:2 * len(coordinates)]

print(x)
print(u)