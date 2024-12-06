# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:37
@Author  : Kend
@FileName: __init__.py.py
@Software: PyCharm
@modifier:
"""
import cv2
import numpy as np

class PolygonMaskProcessor:
    def __init__(self, image_shape, polygon_points):
        """
        初始化多边形掩码处理器，自动选择CPU或GPU处理。

        :param image_shape: 输入图像的形状 (height, width, channels)
        :param polygon_points: 多边形顶点坐标 (list of tuples)
        """
        self.image_height, self.image_width = image_shape[:2]
        self.polygon_points = np.array([polygon_points], dtype=np.int32)

        # 检查CUDA是否可用
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

        if self.use_cuda:
            print("Using GPU for polygon mask processing with async stream.")
            self.stream = cv2.cuda_Stream()
            self._initialize_gpu(image_shape)
        else:
            print("Using CPU for polygon mask processing.")
            self._initialize_cpu(image_shape)

    def _initialize_cpu(self, image_shape):
        """初始化CPU处理所需的资源。"""
        self.mask = self._create_polygon_mask(image_shape)

    def _initialize_gpu(self, image_shape):
        """初始化GPU处理所需的资源。"""
        self.gpu_frame = cv2.cuda_GpuMat()
        self.gpu_mask = cv2.cuda_GpuMat()

        # 提前计算掩码并上传到GPU
        mask = self._create_polygon_mask(image_shape)
        self.gpu_mask.upload(mask)

    def _create_polygon_mask(self, image_shape):
        """创建多边形掩码。"""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        cv2.fillPoly(mask, self.polygon_points, 255)  # 将多边形区域填充为白色（255）
        return mask

    def apply_mask(self, image):
        """
        应用多边形掩码，将多边形外部区域涂成灰色。

        :param image: 输入图像 (numpy array)
        :return: 应用掩码后的图像 (numpy array)
        """
        if self.use_cuda:
            return self._apply_mask_gpu(image)
        else:
            return self._apply_mask_cpu(image)

    def _apply_mask_cpu(self, image):
        """使用CPU应用多边形掩码。"""
        gray_color = 128  # 灰色值
        masked_image = cv2.bitwise_and(image, image, mask=self.mask)
        external_area = cv2.bitwise_not(self.mask)
        masked_image[external_area > 0] = gray_color  # 将多边形外部区域涂成灰色
        return masked_image

    def _apply_mask_gpu(self, image):
        """使用GPU应用多边形掩码。"""
        # 创建一个与输入图像相同大小的灰度图像
        gray_color = 128  # 灰色值
        gray_image = np.full_like(image, gray_color)

        # 将输入图像和灰度图像上传到GPU
        gpu_gray_image = cv2.cuda_GpuMat()
        gpu_gray_image.upload(gray_image, stream=self.stream)
        self.gpu_frame.upload(image, stream=self.stream)

        # 使用位运算将多边形内部保留，外部涂成灰色
        gpu_masked_image = cv2.cuda.bitwise_and(self.gpu_frame, self.gpu_frame, mask=self.gpu_mask, stream=self.stream)
        gpu_external_area = cv2.cuda.bitwise_not(self.gpu_mask, stream=self.stream)
        gpu_final_image = cv2.cuda.add(gpu_masked_image, gpu_gray_image, mask=gpu_external_area, stream=self.stream)

        # 异步等待流中的所有操作完成
        self.stream.waitForCompletion()

        # 将结果从GPU下载到CPU
        final_image = gpu_final_image.download()

        return final_image

# 示例用法
if __name__ == "__main__":
    # 读取输入图像
    image = cv2.imread("input_image.jpg")

    # 定义多边形顶点坐标 (x, y)
    polygon_points = [
        (50, 50),
        (200, 50),
        (250, 200),
        (50, 200)
    ]

    # 初始化多边形掩码处理器，自动选择CPU或GPU
    mask_processor = PolygonMaskProcessor(image.shape, polygon_points)

    # 应用多边形掩码
    masked_image = mask_processor.apply_mask(image)

    # 显示结果
    cv2.imshow("Masked Image", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite("masked_image.jpg", masked_image)