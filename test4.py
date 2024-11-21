import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from skimage.transform import radon, iradon

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 1. 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=1000):
    sigma = var ** 0.1
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return noisy_image

# 2. 算术均值滤波
def arithmetic_mean_filter(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    return cv2.filter2D(image, -1, kernel)

# 3. 几何均值滤波
def geometric_mean_filter(image, kernel_size=5):
    # 将图像转换为浮点数，以避免溢出
    image = image.astype(np.float32)
    # 创建结果图像
    filtered_image = np.zeros_like(image)
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant')

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            # 获取当前窗口
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            # 计算几何均值
            geometric_mean = np.exp(np.mean(np.log(window[window > 0])))  # 只计算大于0的值
            filtered_image[i - pad_size, j - pad_size] = geometric_mean

    return np.clip(filtered_image, 0, 255).astype(np.uint8)


# 4. 自适应局部降噪滤波
def adaptive_local_denoising(image):
    if len(image.shape) == 3:  # 如果是RGB图像
        channels = []
        for i in range(image.shape[2]):
            channels.append(denoise_wavelet(image[:, :, i], convert2ycbcr=False))
        return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
    else:
        return denoise_wavelet(image)

# 5. 添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = np.ceil(salt_prob * total_pixels)
    num_pepper = np.ceil(pepper_prob * total_pixels)

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255  # Salt

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # Pepper

    return noisy_image

# 6. 中值滤波
def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# 7. 修正后的阿尔法均值滤波
def modified_alpha_mean_filter(image, alpha=0.1):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)

# 8. 自适应中值滤波
def adaptive_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# 9. 运动模糊
def motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

# 10. 维纳滤波
def wiener_filter(image):
    if len(image.shape) == 2:  # 如果是灰度图像
        return cv2.fastNlMeansDenoising(image, None, 10, 10, 7)
    else:  # 如果是彩色图像
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 11. 反卷积（示例）
def deconvolution(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# 12. 拉东变换
def radon_transform(image):
    if len(image.shape) == 3:  # 如果是RGB图像，转换为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    padded_image = np.pad(image, ((image.shape[0] // 2, image.shape[0] // 2),
                                  (image.shape[1] // 2, image.shape[1] // 2)),
                          'constant', constant_values=0)

    theta = np.linspace(0., 180., max(padded_image.shape), endpoint=False)
    sinogram = radon(padded_image, theta=theta, circle=True)
    return sinogram, theta

def iradon_transform(sinogram, theta):
    return iradon(sinogram, theta=theta, circle=True)

# 主程序
if __name__ == "__main__":
    # 读取图像
    image = cv2.imread("D:\\python-Digital Image Processing\\test.jpg")
    if image is None:
        print("无法加载图像，请检查路径。")
    else:
        # 转换为黑白图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 高斯噪声退化与复原
        noisy_image = add_gaussian_noise(gray_image)
        arithmetic_restored = arithmetic_mean_filter(noisy_image)
        geometric_restored = geometric_mean_filter(noisy_image)
        adaptive_restored = adaptive_local_denoising(noisy_image)

        # 显示高斯噪声复原结果
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1), plt.imshow(noisy_image, cmap='gray'), plt.title('高斯噪声')
        plt.subplot(2, 2, 2), plt.imshow(arithmetic_restored, cmap='gray'), plt.title('算术均值滤波')
        plt.subplot(2, 2, 3), plt.imshow(geometric_restored, cmap='gray'), plt.title('几何均值滤波')
        plt.subplot(2, 2, 4), plt.imshow(adaptive_restored, cmap='gray'), plt.title('自适应降噪')
        plt.show()

        # 2. 椒盐噪声退化与复原
        noisy_image_sp = add_salt_and_pepper_noise(gray_image)
        median_restored = median_filter(noisy_image_sp)
        modified_alpha_restored = modified_alpha_mean_filter(noisy_image_sp)
        adaptive_median_restored = adaptive_median_filter(noisy_image_sp)

        # 显示椒盐噪声复原结果
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1), plt.imshow(noisy_image_sp, cmap='gray'), plt.title('椒盐噪声图像')
        plt.subplot(2, 2, 2), plt.imshow(median_restored, cmap='gray'), plt.title('中值滤波')
        plt.subplot(2, 2, 3), plt.imshow(modified_alpha_restored, cmap='gray'), plt.title('修正后的阿尔法均值滤波')
        plt.subplot(2, 2, 4), plt.imshow(adaptive_median_restored, cmap='gray'), plt.title('自适应中值滤波')
        plt.show()

        # 3. 运动模糊与复原
        blurred_image = motion_blur(gray_image)
        noisy_blurred_image = add_gaussian_noise(blurred_image, var=65)
        wiener_restored = wiener_filter(noisy_blurred_image)
        deconvolution_restored = deconvolution(noisy_blurred_image)

        # 显示运动模糊复原结果
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1), plt.imshow(noisy_blurred_image, cmap='gray'), plt.title('模糊且带噪声的图像')
        plt.subplot(2, 2, 2), plt.imshow(wiener_restored, cmap='gray'), plt.title('维纳滤波复原')
        plt.subplot(2, 2, 3), plt.imshow(deconvolution_restored, cmap='gray'), plt.title('反卷积复原')
        plt.show()

        # 4. 拉东变换与重建
        sinogram, theta = radon_transform(gray_image)
        reconstructed_image = iradon_transform(sinogram, theta)

        # 显示拉东变换结果
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1), plt.imshow(sinogram, aspect='auto', cmap='gray'), plt.title('正弦图')
        plt.subplot(1, 2, 2), plt.imshow(reconstructed_image, cmap='gray'), plt.title('重建图像')
        plt.show()
