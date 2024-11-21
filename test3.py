import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
# 读取图像并转换为RGB格式
image = cv2.imread("D:\\python-Digital Image Processing\\test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ============ 1. FFT变换与频谱/相位谱显示 ============ #
def show_fft(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    phase_spectrum = np.angle(fshift)

    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(image)
    plt.title('原始图像'), plt.axis('on')

    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('幅度谱'), plt.axis('on')

    plt.subplot(133), plt.imshow(phase_spectrum, cmap='gray')
    plt.title('相位谱'), plt.axis('on')

    plt.show()

show_fft(image)

# ============ 2. 低通滤波器实现与比较 ============ #
def ideal_lowpass_filter(shape, cutoff):
    rows, cols = shape[:2]
    center = (rows // 2, cols // 2)
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, center, cutoff, 1, -1)
    return mask

def gaussian_lowpass_filter(shape, cutoff):
    rows, cols = shape[:2]
    center = (rows // 2, cols // 2)
    x, y = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = np.exp(-(dist ** 2) / (2 * (cutoff ** 2)))
    return mask

def butterworth_lowpass_filter(shape, cutoff, order):
    rows, cols = shape[:2]
    center = (rows // 2, cols // 2)
    x, y = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = 1 / (1 + (dist / cutoff) ** (2 * order))
    return mask

def apply_filter(image, filter_func, *args):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fshift = np.fft.fftshift(np.fft.fft2(gray_image))
    mask = filter_func(gray_image.shape, *args)
    filtered = fshift * mask
    result = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.abs(result)

# 应用低通滤波器
ideal_lp = apply_filter(image, ideal_lowpass_filter, 30)
gaussian_lp = apply_filter(image, gaussian_lowpass_filter, 30)
butterworth_lp = apply_filter(image, butterworth_lowpass_filter, 30, 2)

plt.figure(figsize=(12, 4))
for i, (img, title) in enumerate(zip([ideal_lp, gaussian_lp, butterworth_lp],
                                      ['理想低通滤波', '高斯低通滤波', '巴特沃斯低通滤波'])):
    plt.subplot(1, 3, i + 1), plt.imshow(img, cmap='gray')
    plt.title(title), plt.axis('on')
plt.show()

# ============ 3. 高通滤波器实现与比较 ============ #
def highpass_filter_from_lowpass(lowpass_filter):
    return 1 - lowpass_filter

# 应用高通滤波器
ideal_hp = highpass_filter_from_lowpass(ideal_lowpass_filter(image.shape, 30))
gaussian_hp = highpass_filter_from_lowpass(gaussian_lowpass_filter(image.shape, 30))
butterworth_hp = highpass_filter_from_lowpass(butterworth_lowpass_filter(image.shape, 30, 2))

ideal_hp_result = apply_filter(image, lambda *args: ideal_hp)
gaussian_hp_result = apply_filter(image, lambda *args: gaussian_hp)
butterworth_hp_result = apply_filter(image, lambda *args: butterworth_hp)

plt.figure(figsize=(12, 4))
for i, (img, title) in enumerate(zip([ideal_hp_result, gaussian_hp_result, butterworth_hp_result],
                                      ['理想高通滤波', '高斯高通滤波', '巴特沃斯高通滤波'])):
    plt.subplot(1, 3, i + 1), plt.imshow(img, cmap='gray')
    plt.title(title), plt.axis('on')
plt.show()

# ============ 4. 同态滤波实现 ============ #
def homomorphic_filter(image, gamma_l=0.5, gamma_h=2.0, cutoff=30):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_log = np.log1p(np.array(gray_image, dtype="float"))
    fshift = np.fft.fftshift(np.fft.fft2(image_log))

    rows, cols = gray_image.shape
    center = (rows // 2, cols // 2)
    x, y = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = (gamma_h - gamma_l) * (1 - np.exp(-(dist ** 2) / (2 * (cutoff ** 2)))) + gamma_l

    filtered = fshift * mask
    result = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.expm1(np.abs(result))

homomorphic_result = homomorphic_filter(image)
homomorphic_result = np.clip(homomorphic_result, 0, 255).astype(np.uint8)

plt.imshow(homomorphic_result, cmap='gray')
plt.title('同态滤波'), plt.axis('on')
plt.show()

# ============ 5. 陷波滤波实现 ============ #
def notch_filter(shape, cutoff, notch_centers):
    rows, cols = shape[:2]
    mask = np.ones((rows, cols), np.uint8)
    for center in notch_centers:
        cv2.circle(mask, center, cutoff, 0, -1)
    return mask

notch_centers = [(100, 100), (200, 200)]  # 示例陷波位置
notch = notch_filter(image.shape, 10, notch_centers)

filtered_notch = apply_filter(image, lambda *args: notch)
plt.imshow(filtered_notch, cmap='gray')
plt.title('陷波滤波'), plt.axis('on')
plt.show()
