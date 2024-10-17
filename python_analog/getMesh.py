import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pygmsh

import pyvista as pv


def angle_between_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    v1 = np.array([x1 - x2, y1 - y2])
    v2 = np.array([x3 - x2, y3 - y2])

    len_v1 = np.linalg.norm(v1)
    len_v2 = np.linalg.norm(v2)

    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (len_v1 * len_v2)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    min_angle = min(angle_deg, 180 - angle_deg)

    return min_angle


def triangle_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))


def plot_triangulation(vertices, triangles):
    """
    Функция для отрисовки триангуляции.
    :param vertices: координаты вершин.
    :param triangles: индексы треугольников.
    """
    # Создание объекта триангуляции
    triangulation = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    # Построение триангуляции
    plt.figure(figsize=(8, 8))
    plt.triplot(triangulation, 'bo-', lw=1.5)
    plt.gca().set_aspect('equal')  # Установка равных пропорций осей
    plt.title("Триангуляция")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


# def getMesh(boundary_points):
#     A = {'vertices': boundary_points}
#     B = tr.triangulate(A, 'a350')
#     v = B['vertices']
#     for tri in B['triangles']:
#         a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
#
#         if triangle_area(a, b, c) < 150 or angle_between_points(a, b, c) < 30:
#             index_to_delete = np.where(np.all(B['triangles'] == tri, axis=1))[0]
#
#             B['triangles'] = np.delete(B['triangles'], index_to_delete, axis=0)
#
#     # Вызов функции для отрисовки триангуляции
#     plot_triangulation(B['vertices'], B['triangles'])
#
#     return B['vertices'], B['triangles']


def getMesh(boundary_points):
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(boundary_points,
            mesh_size=1000,
        )

        mesh = geom.generate_mesh()

        points = mesh.points
        print(points.shape)
        points = np.array(points)
        points = np.delete(points, 2, axis=1)


        # Извлечение ячеек (индексы точек, образующих треугольники)
        # print(mesh.cells_dict)
        cells =  np.array(mesh.cells_dict['triangle'])


        # edges = mesh.extract_feature_edges(boundary_edges=True)
        # bpoints = np.delete(np.array(edges.points), 2, axis=1)

    return points, cells



