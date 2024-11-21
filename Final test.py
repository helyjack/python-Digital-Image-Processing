import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, restoration
from skimage.util import random_noise
from skimage.draw import disk

from test1 import gamma_correction

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

class ImageProcessingApp:
    def __init__(self, master):
        self.img1_gray = None
        self.img1 = None
        self.master = master
        self.master.title("图像处理系统")

        # 选择图像按钮
        self.label = tk.Label(master, text="请选择图像文件:")
        self.label.pack()

        self.load_button = tk.Button(master, text="从电脑选择图像", command=self.load_image)
        self.load_button.pack(pady=10)

        # 处理选项
        self.process_frame = tk.Frame(master)
        self.process_frame.pack(pady=10)

        self.point_ops_button = tk.Button(self.process_frame, text="数字图像点运算", command=self.point_operations)
        self.point_ops_button.grid(row=0, column=0)

        self.spatial_filter_button = tk.Button(self.process_frame, text="数字图像空域滤波",
                                               command=self.spatial_filtering)
        self.spatial_filter_button.grid(row=0, column=1)

        self.frequency_filter_button = tk.Button(self.process_frame, text="数字图像频域滤波",
                                                 command=self.frequency_filtering)  # 添加命令
        self.frequency_filter_button.grid(row=0, column=2)

        self.fixed_psf = np.ones((5, 5)) / 25

        self.restoration_button = tk.Button(self.process_frame, text="数字图像复原及重建",
                                            command=self.restoration_reconstruction)
        self.restoration_button.grid(row=0, column=3)

        self.image_path = ""

        # 存储点运算按钮
        self.point_ops_window = None

        # 存储复原及重建按钮
        self.restoration_window = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="选择图像文件",
                                                     filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not self.image_path:
            messagebox.showerror("错误", "未选择任何图像文件")
            return

        self.img1 = cv2.imread(self.image_path)
        if self.img1 is None:
            messagebox.showerror("错误", "无法加载图像")
            return

        plt.imshow(cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("原始图像")
        plt.show()

    def point_operations(self):
        if not hasattr(self, 'img1'):
            messagebox.showerror("错误", "请先加载图像")
            return

        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)

        # 创建新的窗口
        if self.point_ops_window is None:
            self.point_ops_window = tk.Toplevel(self.master)
            self.point_ops_window.title("数字图像点运算")

            operations = [
                ("加法", self.add_operation),
                ("减法", self.subtract_operation),
                ("乘法", self.multiply_operation),
                ("除法", self.divide_operation),
                ("GAMMA运算", self.gamma_correction_operation),
                ("直方图均衡化", self.equalize_histogram),
                ("直方图匹配", self.match_histogram)
            ]

            for i, (text, command) in enumerate(operations):
                btn = tk.Button(self.point_ops_window, text=text, command=command)
                btn.grid(row=i, column=0, padx=10, pady=5)

    def add_operation(self): #去除随机噪声，增亮。
        img_add = cv2.add(self.img1, self.img1)
        img_add = np.clip(img_add, 0, 255).astype(np.uint8)
        self.display_image("加法图像", img_add)

    def subtract_operation(self): #用于对比两张图的差距，例如一张有噪声，另一张没有。
        img_subtract = cv2.subtract(self.img1, self.img1)
        self.display_image("减法图像", img_subtract)

    def multiply_operation(self):#用于图像的放大，可能会模糊。
        img_multiply = cv2.multiply(self.img1, 1.5)
        img_multiply = np.clip(img_multiply, 0, 255).astype(np.uint8)
        self.display_image("乘法图像", img_multiply)

    def divide_operation(self): #用于图像的缩小，可能会模糊。
        img_divide = cv2.divide(self.img1, 2)
        self.display_image("除法图像", img_divide)

    def gamma_correction_operation(self):#用于图像的变暗或变亮。
        gamma = simpledialog.askfloat("输入Gamma值", "请输入Gamma值（必须大于0且大于1是变暗）:", minvalue=0.1)
        #0-1之间的像素点经过归一化后再乘大于1后会更小，导致变暗。
        if gamma is not None:
            img_gamma = gamma_correction(self.img1, gamma)
            self.display_image("Gamma 变换图像", img_gamma)

    def gamma_correction(image, gamma):
        invGamma = 1.0 / gamma
        table = [((i / 255.0) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        return cv2.LUT(image, table)

    '''将图像的亮度分布均匀化，使得每个灰度级都有大致相同的像素数量。
    这样，图像的对比度就会得到增强，使得图像的细节更加清晰。'''
    def equalize_histogram(self):#用于图像的对比度增强。
        channels = cv2.split(self.img1)
        equalized_channels = [cv2.equalizeHist(ch) for ch in channels]
        img_equalized = cv2.merge(equalized_channels)
        self.display_image("均衡化图像", img_equalized)

    '''直方图是图像中每个灰度级出现的频率的统计
    每个像素点进行分等级，然后乘以n-1，取近似值。'''

    def match_histogram(self):#用于图像的对比度增强。将源图像的直方图匹配到目标图像的直方图。
        # 这样，源图像的亮度分布就会与目标图像的亮度分布一致，从而增强图像的对比度。
        img_target = filedialog.askopenfilename(title="选择目标图像",
                                                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if img_target:
            img_target = cv2.imread(img_target)
            if img_target is None:
                messagebox.showerror("错误", "无法加载目标图像")
                return

            matched = exposure.match_histograms(self.img1, img_target, channel_axis=-1)
            self.display_image("直方图匹配图像", matched)
        else:
            messagebox.showerror("错误", "未选择目标图像")
        #分别计算直方图，cdf，然后把源像素映射到目标cdf对应值。
    def spatial_filtering(self):
        if not hasattr(self, 'img1'):
            messagebox.showerror("错误", "请先加载图像")
            return

        # 创建新窗口以选择滤波器
        filter_window = tk.Toplevel(self.master)
        filter_window.title("选择滤波器")

        def apply_filter(filter_name):
            # 先对原始彩色图像self.img1进行滤波操作
            global filtered_image
            '''方框滤波的基本操作是对图像的每个像素及其邻域内的像素进行求和，
            然后除以邻域内所有的像素的数量，得到新的像素值。
            具体来说，对于图像中的一个像素，其邻域内的像素值之和除以邻域内像素的数量，就是该像素的新值。
            方框滤波是一种简单的空间滤波方法，用于平滑（值越大效果越好）图像或减少图像噪声'''
            if filter_name == "Box 3x3":
                filtered_image = cv2.boxFilter(self.img1, -1, (3, 3))
            elif filter_name == "Box 5x5":
                filtered_image = cv2.boxFilter(self.img1, -1, (5, 5))
            elif filter_name == "Box 7x7":
                filtered_image = cv2.boxFilter(self.img1, -1, (7, 7))

            #高斯滤波的基本操作是对图像的每个像素及其邻域内的像素进行加权平均，权重由高斯函数决定。
            # 具体来说，对于图像中的一个像素，其邻域内的像素值乘以高斯函数的值，然后求和，得到新的像素值。
            elif filter_name == "高斯滤波器":
                filtered_image = cv2.GaussianBlur(self.img1, (5, 5), 0)
            elif filter_name == "最大值滤波器":#其邻域内的最大像素值就是该像素的新值
                filtered_image = cv2.dilate(self.img1, np.ones((5, 5)))
            elif filter_name == "最小值滤波器":#其邻域内的最小像素值就是该像素的新值
                filtered_image = cv2.erode(self.img1, np.ones((5, 5)))
            elif filter_name == "中值滤波器":#其邻域内的中值就是该像素的新值
                filtered_image = cv2.medianBlur(self.img1, 5)
            elif filter_name == "拉普拉斯滤波器":#拉普拉斯算子即对某个像素点求二阶偏导，与图像进行卷积操作
                filtered_image = cv2.Laplacian(self.img1, cv2.CV_64F)
            elif filter_name == "Sobel 滤波器":#x下减上，y右减左，减完求和。121.
                sobel_x = cv2.Sobel(self.img1, cv2.CV_64F, 1, 0, ksize=5)
                sobel_y = cv2.Sobel(self.img1, cv2.CV_64F, 0, 1, ksize=5)
                filtered_image = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
            elif filter_name == "Roberts 滤波器":#同sobel
                roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
                roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
                filtered_image = cv2.filter2D(self.img1, -1, roberts_x) + cv2.filter2D(self.img1, -1, roberts_y)
            '''Roberts滤波器：由于卷积核较小，检测到的边缘可能不如Sobel滤波器准确，但计算速度更快。
            Sobel滤波器：由于卷积核较大且权重合理，能够检测到更准确的边缘，但计算速度较慢。

            Roberts滤波器：适用于对实时性要求较高的应用场景，如实时视频处理。
            Sobel滤波器：适用于对边缘检测精度要求较高的应用场景，如图像处理和计算机视觉任务。'''
            # 确保滤波后的图像数据类型为np.uint8（部分滤波器可能改变数据类型）
            if filtered_image.dtype != np.uint8:
                filtered_image = cv2.convertScaleAbs(filtered_image)

            # 这里先判断图像是否为彩色图像（有3个通道）
            if len(filtered_image.shape) == 3 and filtered_image.shape[2] == 3:
                # 对于彩色图像，直接将其从BGR转换为RGB用于显示
                filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

            # 显示滤波结果
            self.display_image(filter_name, filtered_image)

        # 创建按钮选择滤波器
        filters = [
            "Box 3x3", "Box 5x5", "Box 7x7",
            "高斯滤波器", "最大值滤波器", "最小值滤波器",
            "中值滤波器", "拉普拉斯滤波器", "Sobel 滤波器", "Roberts 滤波器"
        ]

        for filter_name in filters:
            button = tk.Button(filter_window, text=filter_name, command=lambda name=filter_name: apply_filter(name))
            button.pack(pady=5)

    def restoration_reconstruction(self):  # 创建新窗口以选择滤波器
        filter_window = tk.Toplevel(self.master)
        filter_window.title("选择滤波器")

    def frequency_filtering(self):
        if not hasattr(self, 'img1'):
            messagebox.showerror("错误", "请先加载图像")
            return

        # 创建新窗口以选择滤波器
        filter_window = tk.Toplevel(self.master)
        filter_window.title("频域滤波")

        # 定义频域滤波操作
        def apply_fft():
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            phase_spectrum = np.angle(fshift)

            plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('频谱图'), plt.xticks([]), plt.yticks([])
            #将信号进行傅里叶变换，得到信号的频域表示。
            #将傅里叶变换的结果绘制成频谱图，其中频率轴表示信号的频率，幅度轴表示信号的幅度
            plt.subplot(122), plt.imshow(phase_spectrum, cmap='gray')
            plt.title('相位谱图'), plt.xticks([]), plt.yticks([])
            #从傅里叶变换的结果中计算信号的相位谱，并将其绘制成相位谱图。
            plt.show()

        def ideal_low_pass():
            d0 = 30  # 固定截止频率
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2

            mask = np.zeros((rows, cols), np.uint8)
            y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
            mask[np.sqrt(x * x + y * y) <= d0] = 1

            fshift_filtered = fshift * mask
            img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            img_filtered = np.abs(img_filtered)

            self.display_image("理想低通滤波", img_filtered)
            #允许低于某一截止频率的信号通过，而高于截止频率的信号被完全阻止。

        def gaussian_low_pass():
            d0 = 30  # 固定截止频率
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2

            x = np.linspace(-ccol, ccol - 1, cols)
            y = np.linspace(-crow, crow - 1, rows)
            x, y = np.meshgrid(x, y)
            gauss_mask = np.exp(-(x ** 2 + y ** 2) / (2 * (d0 ** 2)))

            fshift_filtered = fshift * gauss_mask
            img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            img_filtered = np.abs(img_filtered)

            self.display_image("高斯低通滤波", img_filtered)
            #它允许低于某一截止频率的信号通过，而高于截止频率的信号被逐渐衰减。

        def butterworth_low_pass():
            d0 = 30  # 固定截止频率
            n = 2  # 固定滤波器阶数
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2

            x = np.linspace(-ccol, ccol - 1, cols)
            y = np.linspace(-crow, crow - 1, rows)
            x, y = np.meshgrid(x, y)
            D = np.sqrt(x ** 2 + y ** 2)
            butter_mask = 1 / (1 + (D / d0) ** (2 * n))

            fshift_filtered = fshift * butter_mask
            img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            img_filtered = np.abs(img_filtered)

            self.display_image("巴特沃斯低通滤波", img_filtered)
            #巴特沃斯低通滤波器是一种低通滤波器，它允许低于某一截止频率的信号通过，而高于截止频率的信号被逐渐衰减。

        def ideal_high_pass():
            d0 = 30  # 固定截止频率
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2

            mask = np.ones((rows, cols), np.uint8)
            y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
            mask[np.sqrt(x * x + y * y) <= d0] = 0

            fshift_filtered = fshift * mask
            img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            img_filtered = np.abs(img_filtered)

            self.display_image("理想高通滤波", img_filtered)
            #允许高于某一截止频率的信号通过，而低于截止频率的信号被完全阻止
            #理想高通滤波器是一种简单的高通滤波器，它允许高于某一截止频率的信号通过，而低于截止频率的信号被完全阻止。
        def gaussian_high_pass():
            d0 = 30  # 固定截止频率
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2

            x = np.linspace(-ccol, ccol - 1, cols)
            y = np.linspace(-crow, crow - 1, rows)
            x, y = np.meshgrid(x, y)
            gauss_mask = 1 - np.exp(-(x ** 2 + y ** 2) / (2 * (d0 ** 2)))

            fshift_filtered = fshift * gauss_mask
            img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            img_filtered = np.abs(img_filtered)

            self.display_image("高斯高通滤波", img_filtered)
            #高斯高通滤波器是一种高通滤波器，它允许高于某一截止频率的信号通过，而低于截止频率的信号被逐渐衰减。
        def butterworth_high_pass():
            d0 = 30  # 固定截止频率
            n = 2  # 固定滤波器阶数
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2

            x = np.linspace(-ccol, ccol - 1, cols)
            y = np.linspace(-crow, crow - 1, rows)
            x, y = np.meshgrid(x, y)
            D = np.sqrt(x ** 2 + y ** 2)
            butter_mask = 1 / (1 + (d0 / (D+1e-6)) ** (2 * n))
            butter_mask[D == 0] = 0  # 避免除零错误

            fshift_filtered = fshift * butter_mask
            img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            img_filtered = np.abs(img_filtered)

            self.display_image("巴特沃斯高通滤波", img_filtered)
            #巴特沃斯高通滤波器是一种高通滤波器，它允许高于某一截止频率的信号通过，而低于截止频率的信号被逐渐衰减。

        def notch_filter():
            # 实现陷波滤波，具体参数可根据需求设置
            d0 = 30  # 固定截止频率
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2

            mask = np.ones((rows, cols), np.uint8)
            mask[crow - d0:crow + d0, ccol - d0:ccol + d0] = 0  # 中心区域为0，形成陷波

            fshift_filtered = fshift * mask
            img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            img_filtered = np.abs(img_filtered)

            self.display_image("陷波滤波", img_filtered)
            #陷波滤波器是一种滤波器，它能够消除图像中的特定频率成分，通常用于消除周期性的噪声，如周期性的亮光或暗光条纹。


        # 创建按钮
        tk.Button(filter_window, text="FFT变换", command=apply_fft).pack(pady=5)
        tk.Button(filter_window, text="理想低通滤波", command=ideal_low_pass).pack(pady=5)
        tk.Button(filter_window, text="高斯低通滤波", command=gaussian_low_pass).pack(pady=5)
        tk.Button(filter_window, text="巴特沃斯低通滤波", command=butterworth_low_pass).pack(pady=5)
        tk.Button(filter_window, text="理想高通滤波", command=ideal_high_pass).pack(pady=5)
        tk.Button(filter_window, text="高斯高通滤波", command=gaussian_high_pass).pack(pady=5)
        tk.Button(filter_window, text="巴特沃斯高通滤波", command=butterworth_high_pass).pack(pady=5)
        tk.Button(filter_window, text="陷波滤波", command=notch_filter).pack(pady=5)

    def restoration_reconstruction(self):
        if not hasattr(self, 'img1'):
            messagebox.showerror("错误", "请先加载图像")
            return

        # 创建新的复原及重建窗口
        if self.restoration_window is None:
            self.restoration_window = tk.Toplevel(self.master)
            self.restoration_window.title("数字图像复原及重建")

        operations = [
            ("添加高斯噪声", self.add_gaussian_noise),
            ("算术均值滤波", self.arithmetic_mean_filter),
            ("几何均值滤波", self.geometric_mean_filter),
            ("自适应局部降噪滤波", self.adaptive_local_denoising),
            ("添加椒盐噪声", self.add_salt_and_pepper_noise),
            ("中值滤波", self.median_filter),
            ("修正后的阿尔法均值滤波", self.modified_alpha_mean_filter),
            ("自适应中值滤波", self.adaptive_median_filter),
            ("运动模糊", self.motion_blur),
            ("维纳滤波", self.wiener_filter),
        ]

        for i, (text, command) in enumerate(operations):
            btn = tk.Button(self.restoration_window, text=text, command=command)
            btn.grid(row=i, column=0, padx=10, pady=5)

    def add_gaussian_noise(self):
        mean, var = 0, 1000
        noisy_img = random_noise(self.img1, mode='gaussian', mean=mean, var=var / 255.0)
        self.display_image("添加高斯噪声", (noisy_img * 255).astype(np.uint8))
            #高斯噪声是一种随机噪声，其概率密度函数服从高斯分布。
        # 在图像处理中，高斯噪声通常用于模拟自然环境中常见的噪声，如拍摄图像时的传感器噪声。

    def arithmetic_mean_filter(self):
        kernel_size = 3
        img_filtered = cv2.blur(self.img1, (kernel_size, kernel_size))
        self.display_image("算术均值滤波", img_filtered)
        #算术均值滤波是一种简单的滤波方法，它通过计算图像中每个像素周围像素的平均值来平滑图像。

    def geometric_mean_filter(self):
        img_float = np.float32(self.img1) + 1e-10
        log_img = np.log(img_float)
        img_filtered = np.exp(cv2.blur(log_img, (3, 3)))
        self.display_image("几何均值滤波", img_filtered.astype(np.uint8))
        #几何均值滤波是一种滤波方法，它通过计算图像中每个像素周围像素的几何平均值(乘积开根)来平滑图像。

    def adaptive_local_denoising(self):
        img_denoised = restoration.denoise_bilateral(self.img1, channel_axis=-1)
        self.display_image("自适应局部降噪", (img_denoised * 255).astype(np.uint8))
        #自适应局部降噪是一种滤波方法，它通过计算图像中每个像素周围像素的局部均值和标准差来平滑图像。

    def add_salt_and_pepper_noise(self):
        noisy_img = random_noise(self.img1, mode='s&p', amount=0.1)
        self.display_image("添加椒盐噪声", (noisy_img * 255).astype(np.uint8))
        #椒盐噪声是一种随机噪声，它由黑白像素组成，类似于盐和胡椒的混合。

    def median_filter(self):
        img_filtered = cv2.medianBlur(self.img1, 3)
        self.display_image("中值滤波", img_filtered)
        #中值滤波是一种滤波方法，它通过计算图像中每个像素周围像素的中值来平滑图像。

    def modified_alpha_mean_filter(self):
        alpha = 0.1
        img_float = np.float32(self.img1)
        img_filtered = np.clip(img_float + alpha, 0, 255)
        self.display_image("修正后的阿尔法均值滤波", img_filtered.astype(np.uint8))
        #修正后的阿尔法均值滤波是一种滤波方法，它通过计算图像中每个像素周围像素的修正均值来平滑图像。
        #修正均值是通过将每个像素的值乘以一个修正系数来实现的。
        #修正系数通常是一个小于1的值，用于减少噪声的影响。

    def adaptive_median_filter(self):
        img_filtered = cv2.bilateralFilter(self.img1, 9, 75, 75)
        self.display_image("自适应中值滤波", img_filtered)
        #自适应中值滤波是一种滤波方法，它通过计算图像中每个像素周围像素的自适应中值来平滑图像。

    def motion_blur(self):
        kernel_motion_blur = np.zeros((15, 15))
        kernel_motion_blur[int((15 - 1) / 2), :] = np.ones(15)
        kernel_motion_blur = kernel_motion_blur / 15
        img_blurred = cv2.filter2D(self.img1, -1, kernel_motion_blur)
        self.display_image("运动模糊", img_blurred)
        #运动模糊是一种模拟相机移动时产生的模糊效果。
        #它通过在图像上应用一个运动模糊核来实现。
        #运动模糊核是一个二维矩阵，它定义了图像中每个像素的模糊程度。

    def wiener_filter(self):
        img_filtered = np.zeros_like(self.img1)
        for i in range(self.img1.shape[2]):  # Assuming the image has 3 channels (RGB)
            img_filtered[:, :, i] = restoration.wiener(self.img1[:, :, i], np.ones((5, 5)) / 25, balance=0.01)
        self.display_image("维纳滤波", (img_filtered * 255).astype(np.uint8))
        #维纳滤波是一种频域滤波方法，它通过最小化图像的均方误差来恢复图像。

    def deconvolution(self, psf):
        # Initialize the output image
        img_deconv = np.zeros_like(self.img1)

        # Loop over each color channel (RGB)
        for i in range(self.img1.shape[2]):  # Assuming the image has 3 channels (RGB)
            # Process each channel independently
            img_deconv[:, :, i] = restoration.unsupervised_wiener(self.img1[:, :, i], psf)[0]

        # Return the deconvolved image
        return img_deconv

    def apply_circular_mask(self):
        # 创建一个与图像相同形状的零矩阵作为遮罩
        mask = np.zeros_like(self.img1)
        rr, cc = disk((self.img1.shape[0] // 2, self.img1.shape[1] // 2), min(self.img1.shape) // 2)
        mask[rr, cc] = 1
        # 对当前加载的图像应用遮罩
        self.img1 = self.img1 * mask
        return self.img1


        # 辅助显示函数

    def display_image(self, title, image):
        # 先判断图像是否为彩色图像（有3个通道）
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 如果是彩色图像，确保其数据类型为np.uint8
            if image.dtype != np.uint8:
                image = cv2.convertScaleAbs(image)
            # 将彩色图像从BGR转换为RGB用于显示
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            # 如果是灰度图像，确保其数据类型为np.uint8
            if image.dtype != np.uint8:
                image = cv2.convertScaleAbs(image)

        plt.figure()
        plt.title(title)
        plt.imshow(image)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
