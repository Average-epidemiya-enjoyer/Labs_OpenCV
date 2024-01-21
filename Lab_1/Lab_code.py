import cv2
import numpy as np
import matplotlib.pyplot as plt

# Глобальные переменные
HIST_SIZE = 256
HIST_RANGE = (0, 256)
IMAGE_SIZE = (256, 256)


def read_and_resize_image(image_path, crop_start=(100, 100), crop_end=(500, 500)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Ошибка при чтении изображения: {image_path}")
    cropped_image = image[crop_start[1]:crop_end[1], crop_start[0]:crop_end[0]]
    resized_image = cv2.resize(cropped_image, IMAGE_SIZE)
    return resized_image


image_1 = read_and_resize_image("moon.png")
image_2 = read_and_resize_image("text.jpg")


def plot_image(image, title, subplot_pos):
    plt.subplot(subplot_pos)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)


plt.figure(figsize=(10, 5))
plot_image(image_1, 'Original Image_1', 121)
plot_image(image_2, 'Original Image_2', 122)
plt.show()


def make_histogram(image, title='Цветовая гистограмма'):
    im_BGR = cv2.split(image)
    colors = ['blue', 'green', 'red']
    plt.figure(figsize=(8, 6))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([im_BGR[i]], [0], None, [HIST_SIZE], HIST_RANGE)
        plt.plot(hist, color=color, label=color.capitalize())
    plt.title(title)
    plt.xlabel('Значение пикселя')
    plt.ylabel('Частота')
    plt.legend()
    plt.show()


def adjust_contrast(image, alpha=1.1):
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
    image_BGR = cv2.split(image)
    contrasted_image_BGR = [(np.clip((((layer - layer.min()) / (layer.max() - layer.min())) ** alpha), 0, 1)) for layer
                            in image_BGR]
    contrasted_image = cv2.merge(contrasted_image_BGR)
    if image.dtype == np.uint8:
        contrasted_image = (255 * contrasted_image).clip(0, 255).astype(np.uint8)
    return contrasted_image


make_histogram(image_1)
make_histogram(image_2)


def equalize_histogram(image):
    image_BGR = cv2.split(image)
    equalized_channels = [cv2.equalizeHist(channel) for channel in image_BGR]
    equalized_image = cv2.merge(equalized_channels)
    return equalized_image


def create_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    clahe_image = cv2.merge([l_channel_clahe, a_channel, b_channel])
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_LAB2BGR)
    return clahe_image


def compare_images(original_image, processed_image, titles=('Оригинальное изображение', 'Обработанное изображение')):
    plt.figure(figsize=(12, 6))
    plot_image(original_image, titles[0], 121)
    plot_image(processed_image, titles[1], 122)
    plt.show()


def plot_profile(image, axis='x'):
    if axis == 'x':
        profile = image[round(image.shape[0] / 2), :] / np.max(image)
        color = 'blue'
        title = 'Проекция по x'
    else:
        profile = image[:, round(image.shape[1] / 2)] / np.max(image)
        color = 'red'
        title = 'Проекция по y'
    plt.plot(profile[:, 0], color=color)
    plt.title(title)


# Пример использования равномерного преобразования гистограммы
equalized_image = equalize_histogram(image_2)
compare_images(image_2, equalized_image)

# Пример использования CLAHE
clahe_image = create_clahe(image_2)
compare_images(image_2, clahe_image)

# Пример использования профилей
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plot_profile(image_2, 'x')
plt.subplot(1, 2, 2)
plot_profile(image_2, 'y')
plt.show()

rows_1, cols_1, channels_1 = image_1.shape

image_1_rgb = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
contr_image_1 = adjust_contrast(image_1_rgb, alpha=1.1)

plt.figure()
plt.imshow(cv2.cvtColor(contr_image_1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Изображение с увеличенным контрастом (Image_1)')
plt.show()

make_histogram(contr_image_1)

image_BGR_1 = cv2.split(image_1)
b_hist_1 = cv2.calcHist([image_BGR_1[0]], [0], None, [HIST_SIZE], HIST_RANGE)
g_hist_1 = cv2.calcHist([image_BGR_1[1]], [0], None, [HIST_SIZE], HIST_RANGE)
r_hist_1 = cv2.calcHist([image_BGR_1[2]], [0], None, [HIST_SIZE], HIST_RANGE)

CH_b_1 = np.cumsum(b_hist_1) / (rows_1 * cols_1)
CH_g_1 = np.cumsum(g_hist_1) / (rows_1 * cols_1)
CH_r_1 = np.cumsum(r_hist_1) / (rows_1 * cols_1)
image_new1_1 = np.zeros((rows_1, cols_1, 3), dtype=np.uint8)

for i in range(rows_1):
    for j in range(cols_1):
        image_new1_1[i, j, 0] = ((np.max(image_BGR_1[0]) - np.min(image_BGR_1[0])) * CH_b_1[image_1[i, j, 0]] + np.min(
            image_BGR_1[0])).astype(np.uint8)
        image_new1_1[i, j, 1] = ((np.max(image_BGR_1[1]) - np.min(image_BGR_1[1])) * CH_g_1[image_1[i, j, 1]] + np.min(
            image_BGR_1[1])).astype(np.uint8)
        image_new1_1[i, j, 2] = ((np.max(image_BGR_1[2]) - np.min(image_BGR_1[2])) * CH_r_1[image_1[i, j, 2]] + np.min(
            image_BGR_1[2])).astype(np.uint8)

plt.figure()
plt.imshow(cv2.cvtColor(image_new1_1, cv2.COLOR_BGR2RGB))
plt.title('Изображение после равномерного преобразования гистограммы (Image_1)')
plt.axis('off')
plt.show()

make_histogram(image_new1_1)


def plot_barcode_profile(image_path):
    barcode_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if barcode_image is None:
        raise ValueError(f"Ошибка при чтении изображения: {image_path}")

    row_index = barcode_image.shape[0] // 2
    profile = barcode_image[row_index, :]

    plt.figure()
    plt.plot(profile, color='green')
    plt.title('Профиль по x для изображения штрих-кода')
    plt.xlabel('Позиция пикселя')
    plt.ylabel('Значение пикселя')
    plt.show()


image_path = 'barcode.png'
plot_barcode_profile(image_path)
