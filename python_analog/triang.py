import numpy as np
from scipy.spatial import Delaunay
import triangle
def generate_internal_points(vertices):
    # Функция, которая генерирует точки внутри многоугольника
    # Можно использовать различные алгоритмы для этого
    # Один из возможных способов - регулярная сетка внутри ограничивающего многоугольника
    # Например, можно использовать метод перебора всех точек внутри многоугольника и выбирать их
    # или использовать метод случайных точек с последующей фильтрацией
    # В данном примере, просто выберем регулярную сетку точек внутри ограничивающего многоугольника

    # Найдем минимальные и максимальные значения по осям
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])

    # Генерируем сетку точек внутри многоугольника
    step = 25  # Шаг сетки
    points = []
    for x in np.arange(min_x + step, max_x, step):
        for y in np.arange(min_y + step, max_y, step):
            # Проверяем, находится ли точка внутри многоугольника
            if is_point_inside_polygon(x, y, vertices):
                points.append([x, y])
            print(x, y )

    return np.array(points)

def is_point_inside_polygon(x, y, vertices):
    # Функция проверки, находится ли точка внутри многоугольника
    # Используется алгоритм проверки на четность числа пересечений луча,
    # исходящего из точки и параллельного оси x, с границами многоугольника

    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

# Пример данных - границы невыпуклого тела
# boundary_points = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1.5], [0, 1]])
boundary_points = np.loadtxt('P.dat', dtype=float)

# Генерация внутренних точек
internal_points = generate_internal_points(boundary_points)

# Совмещение граничных точек и внутренних точек
all_points = np.concatenate([boundary_points, internal_points])

# Триангуляция
#tri = Delaunay(all_points, incremental=True, qhull_options="QJ")
# tri = triangle.delaunay(all_points)

tri = triangle.triangulate({'vertices': boundary_points}, 'a100')
print(tri)
# Визуализация результатов (может потребоваться установка библиотек для графики, например, matplotlib)
# import matplotlib.pyplot as plt
#
# # plt.triplot(all_points[:,0], all_points[:,1], tri['triangles'])
# plt.plot(tri['vertices'][:,0], tri['vertices'][:,1], 'o')
# # plt.plot(boundary_points[:,0], boundary_points[:,1], 'o')
# plt.show()
