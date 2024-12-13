#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 15:12
@Description: preprocess_image - 数据预处理
@Modify:
@Contact: tankang0722@gmail.com
"""
import time

import numpy as np
import cv2


def preprocess_image(image, input_size, mean=None, std=None, swap=(2, 0, 1)):
    """
    神经网络前预处理图像
    image: 输入的原始图像数据。
    input_size: 模型要求的输入尺寸，是一个二维元组 (height, width)。
    mean: 图像归一化使用的均值，可以是单个值或一个列表/元组，对应每个通道的均值。
    std: 图像归一化使用的标准差，可以是单个值或一个列表/元组，对应每个通道的标准差。
    swap: 图像通道顺序的调整，默认值为 (2, 0, 1)，意味着从 (height, width, channels) 转换到 (channels, height, width)。
    Return : 返回处理好的图像以及计算出的缩放比例 r，r后者可用于后续的结果恢复，例如检测框的坐标调整.
    """
    # 创建填充图像：
    # rgb通道填充114.0，灰度图填充114.0
    # cv2.imshow("img1", image)
    # cv2.waitKey(0)
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0

    # 调整图像大小并填充到指定尺寸：
    img = np.array(image)
    # print(img.shape)  # (720, 1280, 3)  原图
    # cv2.imshow("img1", padded_img)
    # cv2.waitKey(0)
    # 计算缩放比例
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])  # 计算原图最长边到binding box的比例
    # print(r) # 0.5
    # NOTE : 缩放有问题  -> 是因为np.float32的问题 我们将像素值归一化到0到1之间，这样图像才能正确显示。还需要居中保持
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR,
                             ).astype(np.float32)
    # 将调整后的图像放置到填充图像中：
    # 计算居中的位置
    x_offset = (input_size[1] - int(img.shape[1] * r)) // 2  # 宽度
    y_offset = (input_size[0] - int(img.shape[0] * r)) // 2   # 高度
    # # 将缩放后的图像粘贴到新的空白图像上
    padded_img[y_offset:y_offset + int(img.shape[0] * r),  x_offset:x_offset + int(img.shape[1] * r)] = resized_img
    # padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img  # 没有居中

    # print(padded_img.shape) # (640, 640, 3)
    # padded_img /= 255.0
    # cv2.namedWindow("resized_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("resized_img", padded_img)
    # cv2.waitKey(0)

    # im = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

    # 转置 BGR 2 RGB
    padded_img = padded_img[:, :, ::-1]
    # 归一化
    padded_img /= 255.0
    #  根据提供的均值和标准差对图像进行进一步的归一化处理，最好与目标检测的训练参数保持一致。# NOTE : yolov5 yolo11 默认是NONE
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    # 调整通道顺序  HWC to CHW
    padded_img = padded_img.transpose(swap)
    # 确保数据在内存中是连续存储的
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    # 返回预处理后的图像和缩放比例
    return padded_img, r




""" 在图像预处理之前处理 优点： 后续的检测、跟踪等算法可以专注于多边形内的区域，减少计算量和误检率，提高事件准确率 """
class PolygonMaskProcessor:

    def __init__(self, image_shape, polygon_points=None):
        """
        初始化多边形掩码处理器，自动选择CPU或GPU处理。
        :param image_shape: 输入图像的形状 (height, width, channels)
        :param polygon_points: 多边形顶点坐标 (list of tuples)
        """
        self.image_height, self.image_width = image_shape[:2]
        self.polygon_points = np.array([polygon_points], dtype=np.int32)

        # 检查CUDA是否可用 使用torch.cuda.is_available()速度慢
        # print("是否可用gpu", cv2.cuda.getCudaEnabledDeviceCount())
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0


        if self.use_cuda:
            print("使用GPU处理图像")
            self._initialize_gpu(image_shape)
            self.stream = cv2.cuda_Stream()  # 创建一个流对象
            self._initialize_gpu(image_shape)
        else:
            print("使用cpu处理图像")
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

        # 异步等待流中的所有操作完成，确保当前帧的处理已经完成
        self.stream.waitForCompletion()
        # 将结果从GPU下载到CPU
        final_image = gpu_final_image.download()
        return final_image



if __name__ == '__main__':
    # img_path = r"../../tests/frame_0000.jpg"
    # image = cv2.imread(img_path)
    # pre_img, _ = preprocess_image(image, input_size=(1280, 1280))
    # TODO 需要对比cpu和gpu的速度，以及资源的开销
    # 掩码ROI示例用法
    # 读取输入图像
    time1 = time.time()
    image = cv2.imread("../../tests/frame_0000.jpg")
    print(image.shape)
    # 定义多边形顶点坐标 (x, y)
    polygon_points = [
        (50, 50),
        (200, 50),
        (250, 200),
        (50, 900),
        (600, 600)
    ]

    # 初始化多边形掩码处理器
    mask_processor = PolygonMaskProcessor(image.shape, polygon_points)
    # 应用多边形掩码
    masked_image = mask_processor.apply_mask(image)
    print(masked_image.shape, (time.time()-time1) * 1000 ,"ms")  # a4000 40ms

    # 显示结果
    # cv2.imshow("Masked Image", masked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 保存结果
    # cv2.imwrite("masked_image.jpg", masked_image)