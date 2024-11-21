import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# 辅助函数：显示图像
def display_image(title, image):
    plt.figure()
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为 RGB 以便 Matplotlib 正确显示
    plt.axis('off')
    plt.show()

# 辅助函数：Gamma 校正
def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = [((i / 255.0) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(image, table)

# 1. 加载原始 RGB 图像
img1 = cv2.imread("D:\\python-Digital Image Processing\\test.jpg")

# 确保图像正确加载
if img1 is None:
    raise FileNotFoundError("无法加载原始图像，请检查路径是否正确。")

# 2. 显示原始图像
display_image("Original Image", img1)

# 3. 加法
img_add = cv2.add(img1, img1)
display_image("Added Image", img_add)

# 4. 减法
img_subtract = cv2.subtract(img1, img1)
display_image("Subtracted Image", img_subtract)

# 5. 乘法
img_multiply = cv2.multiply(img1, 1.5)
img_multiply = np.clip(img_multiply, 0, 255).astype(np.uint8)  # 防止溢出
display_image("Multiplied Image", img_multiply)

# 6. 除法
img_divide = cv2.divide(img1, 2)
display_image("Divided Image", img_divide)

# 7. Gamma 变暗
img_gamma_dark = gamma_correction(img1, 2.0)#大于1是变暗
display_image("Gamma Dark Image", img_gamma_dark)

# 8. Gamma 变亮
img_gamma_bright = gamma_correction(img1, 0.5)
display_image("Gamma Bright Image", img_gamma_bright)

# 9. 直方图均衡化（对每个通道分别进行）
channels = cv2.split(img1)
equalized_channels = [cv2.equalizeHist(ch) for ch in channels]
img_equalized = cv2.merge(equalized_channels)
display_image("Equalized Image", img_equalized)

# 10. 直方图匹配
img_target = cv2.imread("D:\\_20240530170147.jpg")

# 检查目标图像是否成功加载
if img_target is None:
    raise FileNotFoundError("无法加载目标图像，请检查路径是否正确。")

# 使用 match_histograms 对 RGB 图像进行匹配
matched = exposure.match_histograms(img1, img_target, channel_axis=-1)
display_image("Histogram Matched Image", matched)

print("图像处理完成。")
