import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体为黑体，解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def apply_filters(image):
    # 将图像从 BGR 转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用不同类型的滤波器
    # Box 滤波器
    box_3x3 = cv2.boxFilter(image_rgb, -1, (3, 3))
    box_5x5 = cv2.boxFilter(image_rgb, -1, (5, 5))
    box_7x7 = cv2.boxFilter(image_rgb, -1, (7, 7))

    # 高斯滤波器
    gaussian = cv2.GaussianBlur(image_rgb, (5, 5), 0)

    # 最大值滤波器
    max_filter = cv2.dilate(image_rgb, np.ones((5, 5)))

    # 最小值滤波器
    min_filter = cv2.erode(image_rgb, np.ones((5, 5)))

    # 中值滤波器
    median_filter = cv2.medianBlur(image_rgb, 5)

    # 拉普拉斯滤波器
    laplacian = cv2.Laplacian(image_rgb, cv2.CV_64F)

    # Sobel 滤波器（梯度算子）
    sobel_x = cv2.Sobel(image_rgb, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image_rgb, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Roberts 交叉算子
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    roberts_combined = cv2.filter2D(image_rgb, -1, roberts_x) + cv2.filter2D(image_rgb, -1, roberts_y)

    # 将结果裁剪到 [0, 255] 并转换为 uint8
    results = [
        image_rgb,
        box_3x3, box_5x5, box_7x7,
        gaussian,
        max_filter,
        min_filter,
        median_filter,
        laplacian,
        sobel_combined,
        roberts_combined
    ]

    results = [np.clip(result, 0, 255).astype(np.uint8) for result in results]

    # 显示滤波结果
    plt.figure(figsize=(15, 10))
    titles = ['原始图像', 'Box 滤波器 3x3', 'Box 滤波器 5x5', 'Box 滤波器 7x7',
              '高斯滤波器', '最大值滤波器', '最小值滤波器', '中值滤波器',
              '拉普拉斯滤波器', 'Sobel 滤波器', 'Roberts 滤波器']

    for i in range(len(results)):
        plt.subplot(4, 3, i + 1)
        plt.imshow(results[i] if i < 9 else results[i], cmap='gray' if i >= 9 else None)
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()  # 显示结果


# 主函数，读取图像并应用滤波器
def main():
    # 读取图像，采用 BGR 模式
    image_path = "D:\\python-Digital Image Processing\\test.jpg"
    image = cv2.imread(image_path)

    if image is None:  # 检查图像是否成功读取
        print("错误：未找到图像。")
        return

    apply_filters(image)  # 调用滤波器应用函数


if __name__ == "__main__":
    main()  # 运行主函数
