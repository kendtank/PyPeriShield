# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/27 下午8:34
@Author  : Kend
@FileName: predictor_yolo11.py
@Software: PyCharm
@modifier:
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os



class PredictorYolo11:
    def __init__(self, ckpt_path, input_size=(640, 640), fp16=False, iou_type=None, conf_thres=0.1, iou_thres=0.4):
        self.iou_type = iou_type  # yolov5的iou_type, 这里是为了保持参数一至
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.txt = True if os.path.basename(ckpt_path).endswith("engine") else False
        # logger.info(f"Loading model {ckpt_path}")
        if self.txt:
            x = torch.ones((1, 3, input_size[0], input_size[1]), device=self.device)
            self.model = YOLO(ckpt_path)
            self.model(x)
            # logger.info("TensorRT engine loaded successfully")
        else:
            self.model = YOLO(ckpt_path)
        # 设置输入图像的尺寸
        self.input_size = input_size
        # 是否使用FP16精度
        self.fp16 = fp16

    def predict(self, image: np.ndarray, timer=None):
        # 获取图像的高度和宽度 图像缩放比例
        ratio = min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
        if timer is not None:
            # 启动计时器
            timer.start()
        # 初始化图像信息字典 # 计算图像缩放比例
        # img_info = {"id": 0, "file_name": None, "height": height, "width": width, "raw_img": image,
        #             "ratio": min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])}
        tensorboard_data = None
        # --------推理---------
        # 使用YOLO11模型进行预测
        results_list = self.model.predict(
            source=image,
            task='detect',
            conf=self.conf_thres,
            half=self.fp16,
            iou=self.iou_thres
        )
        # 遍历检测结果
        for results in results_list:  # 这是一个ultralytics.engine.results.Results对象
            # 获取检测结果的边界框数据
            tensorboard_data = results.boxes.data  # 推理的原始张量的numpy形式的data数据
        return tensorboard_data, ratio



if __name__ == '__main__':
    # 设置当前工作目录为项目根目录
    # /home/lyh/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
    project_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(project_root)
    # 测试图像路径
    image = "tests/frame_0000.jpg"
    # 模型路径
    model_path = "weights/yolo11n.pt"
    # 读取图像
    img = cv2.imread(image)
    # 创建PredictorYolo11实例
    predictor_yolo11 = PredictorYolo11(model_path)
    # 进行预测
    resource, info = predictor_yolo11.predict(img)
    # 打印预测结果和图像信息
    print(resource.shape, info)  # torch.Size([29, 6]) 0.5
