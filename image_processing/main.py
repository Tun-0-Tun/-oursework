import cv2
import numpy as np
import imageio.v2 as imageio


def find_contour(image_path, pic_num):
    # Чтение изображения
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    multi_layer_tiff = imageio.imread(image_path)

    image = np.array(multi_layer_tiff[pic_num], dtype=np.uint8)
    image[image == 0] = 255
    image[image == 1] = 0

    # Поиск контура
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Выбираем наибольший контур
    contour = max(contours, key=cv2.contourArea)

    # Преобразование контура в упорядоченный массив точек
    contour = np.squeeze(contour)

    # Сортировка точек по полярному углу относительно центра масс контура
    center, _ = cv2.minEnclosingCircle(contour)
    contour_sorted = sorted(contour, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))

    return contour_sorted, contours


def draw_contour(image_path, contour, pic_num):
    # Чтение изображения
    # image = cv2.imread(image_path)

    multi_layer_tiff = imageio.imread(image_path)

    matrix = np.array(multi_layer_tiff[pic_num], dtype=np.uint8)
    image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
    image[np.where(matrix == 0)] = [255, 255, 255]
    image[np.where(matrix == 1)] = [0, 0, 0]

    # Отрисовка контура
    cv2.drawContours(image, contour, -1, (0, 255, 0), 1)
    cv2.namedWindow('Contour')
    cv2.setMouseCallback('Contour', mouse_callback, {'image': image})

    # Вывод изображения с контуром
    cv2.imshow('Contour', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = params['image']
        pixel_value = img[y, x]
        print(f"Pixel value at ({x}, {y}): {pixel_value}")


if __name__ == "__main__":
    image_path = "../python_analog/Series015_body.tif"
    contour, cnt = find_contour(image_path, 15)
    print("Контур:", list(map(list, contour)))

    draw_contour(image_path, cnt, 15)
