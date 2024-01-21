import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def load_and_resize_image(image_path, size=(300, 300)):
    image = cv2.imread('cat.jpeg')
    if image is None:
        raise ValueError(f"Ошибка при чтении изображения: {image_path}")
    resized_image = cv2.resize(image, size)
    return resized_image


def display_image(image, title="Изображение"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')


def compare_images(image1, image2, title1="Оригинальное изображение", title2="Результат"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    display_image(image1, title1)
    plt.subplot(1, 2, 2)
    display_image(image2, title2)
    plt.show()


def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows))


def reflect_image(image, axis=0):
    return cv2.flip(image, axis)


def scale_image(image, fx, fy):
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)


def rotate_image(image, angle, center=None, scale=1.0):
    rows, cols = image.shape[:2]
    if center is None:
        center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (cols, rows))


def affine_transform(image, src_points, dst_points):
    M = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))
    rows, cols = image.shape[:2]
    return cv2.warpAffine(image, M, (cols, rows))


def perspective_transform(image, src_points, dst_points):
    M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    rows, cols = image.shape[:2]
    return cv2.warpPerspective(image, M, (cols, rows))


def sinusoidal_distortion(image, amplitude, frequency):
    rows, cols = image.shape[:2]
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    u = u.astype(np.float32) + amplitude * np.sin(2 * np.pi * v.astype(np.float32) / frequency)
    v = v.astype(np.float32)
    return cv2.remap(image, u, v, cv2.INTER_LINEAR)


def barrel_distortion(image, distortion_strength=0.0005):
    rows, cols = image.shape[:2]
    # Координатная сетка для исходного изображения
    x = np.linspace(-cols / 2, cols / 2, cols)
    y = np.linspace(-rows / 2, rows / 2, rows)
    x, y = np.meshgrid(x, y)

    # Радиальное расстояние от центра изображения
    r = np.sqrt(x ** 2 + y ** 2)

    # Бочкообразное искажение
    k = distortion_strength  # Коэффициент искажения
    x_distorted = x * (1 + k * r ** 2)
    y_distorted = y * (1 + k * r ** 2)

    # Приведение координат обратно к диапазону изображения
    x_distorted = (x_distorted + cols / 2).astype(np.float32)
    y_distorted = (y_distorted + rows / 2).astype(np.float32)

    # Применение искажения к изображению
    distorted_image = cv2.remap(image, x_distorted, y_distorted, cv2.INTER_LINEAR)
    return distorted_image


image_path = "cat.jpeg"
original_image = load_and_resize_image(image_path)
translated_image = translate_image(original_image, tx=50, ty=100)
reflected_image = reflect_image(original_image, axis=1)
scaled_image = scale_image(original_image, fx=2, fy=2)
rotated_image = rotate_image(original_image, angle=17)
affine_transformed_image = affine_transform(original_image, src_points=[(50, 50), (200, 50), (50, 200)],
                                            dst_points=[(10, 100), (200, 50), (100, 250)])
perspective_transformed_image = perspective_transform(original_image,
                                                      src_points=[(0, 0), (300, 0), (300, 300), (0, 300)],
                                                      dst_points=[(0, 0), (300, 0), (280, 300), (20, 300)])
sinusoidal_image = sinusoidal_distortion(original_image, amplitude=20, frequency=30)
barrel_distorted_image = barrel_distortion(original_image, distortion_strength=0.0005)

# Сравнение оригинального изображения с преобразованными
compare_images(original_image, translated_image, "Оригинальное изображение", "Сдвиг")
compare_images(original_image, reflected_image, "Оригинальное изображение", "Отраженное изображение")
compare_images(original_image, scaled_image, "Оригинальное изображение", "Масштабирование")
compare_images(original_image, rotated_image, "Оригинальное изображение", "Поворот")
compare_images(original_image, affine_transformed_image, "Оригинальное изображение", "Афинное преобразование")
compare_images(original_image, perspective_transformed_image, "Оригинальное изображение", "Perspective Transformed")
compare_images(original_image, sinusoidal_image, "Оригинальное изображение", "Синусоидальная дисторсия")
compare_images(original_image, barrel_distorted_image, "Оригинальное изображение", "Бочкообразное искажение")


def split_image(image):
    # Функция для разделения изображения на верхнюю и нижнюю половину
    height = image.shape[0]
    top_part = image[:height // 2, :]
    bottom_part = image[height // 2:, :]
    return top_part, bottom_part


def show_split_images(top_part, bottom_part):
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(top_part, cv2.COLOR_BGR2RGB))
    plt.title('Верхняя часть')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(bottom_part, cv2.COLOR_BGR2RGB))
    plt.title('Нижняя часть')
    plt.axis('off')
    plt.show()


def stitch_images(top_part, bottom_part):
    template_size = 10
    template = top_part[-template_size:, :]
    result = cv2.matchTemplate(bottom_part, template, cv2.TM_CCOEFF)
    _, _, _, max_loc = cv2.minMaxLoc(result)


    combined_height = top_part.shape[0] + bottom_part.shape[0] - max_loc[1] - template_size
    combined_image = np.zeros((combined_height, top_part.shape[1], top_part.shape[2]), dtype=np.uint8)
    combined_image[:top_part.shape[0], :] = top_part
    combined_image[top_part.shape[0]:, :] = bottom_part[max_loc[1] + template_size:, :]

    return combined_image


def show_combined_image(combined_image):
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title('Склеенное изображение')
    plt.axis('off')
    plt.show()


top_part, bottom_part = split_image(original_image)
show_split_images(top_part, bottom_part)
combined_image = stitch_images(top_part, bottom_part)
show_combined_image(combined_image)