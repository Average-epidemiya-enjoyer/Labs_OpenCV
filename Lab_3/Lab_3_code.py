import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(image, max_width=500):
    height, width = image.shape[:2]
    if width > max_width:
        scaling_factor = max_width / float(width)
        return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image


image_path = "map.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_1 = resize_image(original_image)
b, g, r = cv2.split(image_1)


# Функции для добавления различных типов шума
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    d = 0.05
    s_vs_p = salt_prob / (salt_prob + pepper_prob)
    rng = np.random.default_rng()
    noise = rng.random(image_1.shape)
    image_noisy = np.copy(image_1)
    image_noisy[noise < d * s_vs_p] = 255
    image_noisy[np.logical_and(noise >= d * s_vs_p, noise < d)] = 0

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(image_noisy)
    plt.title('Шум соль и перец')

    plt.show()
    return image_noisy


def add_gaussian_noise(image, mean, var):
    rng = np.random.default_rng()
    gauss = rng.normal(mean, var ** 0.5, image_1.shape)
    gauss = gauss.reshape(image_1.shape)

    if image_1.dtype == np.uint8:
        gaussian_noisy_image = (image_1.astype(np.float32) + gauss * 255).clip(0, 255).astype(np.uint8)
    else:
        gaussian_noisy_image = (image_1 + gauss).astype(np.float32)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(gaussian_noisy_image)
    plt.title('Гауссовский шум')

    plt.show()
    return gaussian_noisy_image


def add_poisson_noise(image, noise_level):
    rng = np.random.default_rng()

    if image_1.dtype == np.uint8:
        image_p = image_1.astype(np.float32) / 255
        vals = len(np.unique(image_p))
        vals = 2 ** np.ceil(np.log2(vals))
        poisson_noise_image = (255 * (rng.poisson(image_p * vals * noise_level) / float(vals)).clip(0, 1)).astype(
            np.uint8)
    else:
        vals = len(np.unique(image_1))
        vals = 2 ** np.ceil(np.log2(vals))
        poisson_noise_image = rng.poisson(image_1 * vals * noise_level) / float(vals)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(poisson_noise_image)
    plt.title('Пуассоновский шум')

    plt.show()
    return poisson_noise_image


def apply_gaussian_blur(image, sigma):
    sigma = 1.0
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(blurred_image)
    plt.title('Размытие')

    plt.show()
    return blurred_image


def contraharmonic_mean_filter(image, mask, Q):
    poisson_noisy_image = add_poisson_noise(image_1, 0.5)
    image_with_noise = poisson_noisy_image
    padded_image = cv2.copyMakeBorder(image_with_noise, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    result_image = np.zeros_like(image_with_noise, dtype=np.float64)

    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            neighborhood = padded_image[i - 1:i + 2, j - 1:j + 2] * mask
            numerator = np.sum(np.power(neighborhood, Q + 1))
            denominator = np.sum(np.power(neighborhood, Q))
            result_image[i - 1, j - 1] = numerator / denominator
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title('Контргармонический фильтр')

    plt.show()
    return result_image.astype(np.uint8)


def apply_median_filter(image, ksize):
    median_filtered_image = cv2.medianBlur(image, ksize)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(median_filtered_image)
    plt.title('Медианный фильтр')

    plt.show()
    return median_filtered_image


def apply_weighted_median_filter(image, kernel_size, weights):
    assert weights.shape == (kernel_size, kernel_size), "Weights array must be of shape (kernel_size, kernel_size)"

    d = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, d, d, d, d, cv2.BORDER_REFLECT)
    result_image = np.zeros_like(image)

    for i in range(d, padded_image.shape[0] - d):
        for j in range(d, padded_image.shape[1] - d):
            window = padded_image[i - d:i + d + 1, j - d:j + d + 1]
            weighted_window = np.copy(window).astype(np.float32)
            for k in range(kernel_size):
                for l in range(kernel_size):
                    weighted_window[k, l] *= weights[k, l]
            flat_weighted_window = weighted_window.flatten()
            median_value = np.median(flat_weighted_window)
            result_image[i - d, j - d] = median_value
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title('Взвешенный медианный фильтр')

    plt.show()
    return result_image.astype(np.uint8)


def apply_roberts_filter(image):
    b, g, r = cv2.split(image_1)

    G_x = np.array([[1, 0], [0, -1]])
    G_y = np.array([[0, 1], [-1, 0]])

    r_x = cv2.filter2D(r, -1, G_x).astype(np.float32)
    r_y = cv2.filter2D(r, -1, G_y).astype(np.float32)

    g_x = cv2.filter2D(g, -1, G_x).astype(np.float32)
    g_y = cv2.filter2D(g, -1, G_y).astype(np.float32)

    b_x = cv2.filter2D(b, -1, G_x).astype(np.float32)
    b_y = cv2.filter2D(b, -1, G_y).astype(np.float32)

    r_out = cv2.magnitude(r_x, r_y)
    g_out = cv2.magnitude(g_x, g_y)
    b_out = cv2.magnitude(b_x, b_y)

    cv2.normalize(r_out, r_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(g_out, g_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(b_out, b_out, 0, 1, cv2.NORM_MINMAX)

    result_image = cv2.merge([r_out, g_out, b_out])
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title('Фильтр Робертса')

    plt.show()
    return result_image


def apply_prewitt_filter(image):
    b, g, r = cv2.split(image_1)
    G_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    G_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    r_x = cv2.filter2D(r, -1, G_x).astype(np.float32)
    r_y = cv2.filter2D(r, -1, G_y).astype(np.float32)

    g_x = cv2.filter2D(g, -1, G_x).astype(np.float32)
    g_y = cv2.filter2D(g, -1, G_y).astype(np.float32)

    b_x = cv2.filter2D(b, -1, G_x).astype(np.float32)
    b_y = cv2.filter2D(b, -1, G_y).astype(np.float32)

    r_out = cv2.magnitude(r_x, r_y)
    g_out = cv2.magnitude(g_x, g_y)
    b_out = cv2.magnitude(b_x, b_y)

    cv2.normalize(r_out, r_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(g_out, g_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(b_out, b_out, 0, 1, cv2.NORM_MINMAX)

    result_image = cv2.merge([r_out, g_out, b_out])
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title('Фильтр Прюита')

    plt.show()
    return result_image


def apply_sobel_filter(image):
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    r_x = cv2.filter2D(r, -1, G_x).astype(np.float32)
    r_y = cv2.filter2D(r, -1, G_y).astype(np.float32)

    g_x = cv2.filter2D(g, -1, G_x).astype(np.float32)
    g_y = cv2.filter2D(g, -1, G_y).astype(np.float32)

    b_x = cv2.filter2D(b, -1, G_x).astype(np.float32)
    b_y = cv2.filter2D(b, -1, G_y).astype(np.float32)

    r_out = cv2.magnitude(r_x, r_y)
    g_out = cv2.magnitude(g_x, g_y)
    b_out = cv2.magnitude(b_x, b_y)

    cv2.normalize(r_out, r_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(g_out, g_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(b_out, b_out, 0, 1, cv2.NORM_MINMAX)

    result_image = cv2.merge([r_out, g_out, b_out])
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title('Фильтр Собеля')

    plt.show()
    return result_image


def apply_laplacian_filter(image):
    laplas_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    r_laplas = cv2.filter2D(r, -1, laplas_kernel).astype(np.float32)
    g_laplas = cv2.filter2D(g, -1, laplas_kernel).astype(np.float32)
    b_laplas = cv2.filter2D(b, -1, laplas_kernel).astype(np.float32)

    r_out = np.abs(r_laplas)
    g_out = np.abs(g_laplas)
    b_out = np.abs(b_laplas)

    cv2.normalize(r_out, r_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(g_out, g_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(b_out, b_out, 0, 1, cv2.NORM_MINMAX)

    result_image = cv2.merge([r_out, g_out, b_out])
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title('Фильтр Лапласа')

    plt.show()
    return result_image


def apply_canny_filter(image, t1, t2):
    canny = cv2.Canny(image_1, t1, t2)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(canny)
    plt.title('Алгоритм Кэнни')

    plt.show()
    return canny


image_path = "map.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
resized_image = resize_image(original_image)

# Применяем шумы и фильтры к измененному изображению и сохраняем результаты
salt_pepper_noisy = add_salt_and_pepper_noise(resized_image.copy(), salt_prob=0.01, pepper_prob=0.01)
gaussian_noisy = add_gaussian_noise(resized_image.copy(), mean=0, var=0.01)
poisson_noisy = add_poisson_noise(resized_image.copy(), noise_level=30)

# Применяем размытие и фильтры к зашумленным изображениям
gaussian_blurred = apply_gaussian_blur(poisson_noisy.copy(), sigma=1.0)
median_filtered = apply_median_filter(poisson_noisy.copy(), ksize=3)
weighted_median_filtered = apply_weighted_median_filter(poisson_noisy.copy(), kernel_size=3, weights=np.ones((3, 3)) / 9)

# Применяем фильтры обнаружения краев к оригинальному изображению
roberts_edges = apply_roberts_filter(resized_image.copy())
prewitt_edges = apply_prewitt_filter(resized_image.copy())
sobel_edges = apply_sobel_filter(resized_image.copy())
laplacian_edges = apply_laplacian_filter(resized_image.copy())
canny_edges = apply_canny_filter(resized_image.copy(), t1=100, t2=200)

# Списки оригинальных и обработанных изображений для отображения пар
original_images = [resized_image] * 11
processed_images = [
    salt_pepper_noisy, gaussian_noisy, poisson_noisy,
    gaussian_blurred, median_filtered, weighted_median_filtered,
    roberts_edges, prewitt_edges, sobel_edges, laplacian_edges, canny_edges
]
