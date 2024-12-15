# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/27 下午8:34
@Author  : Kend
@FileName: invasion.py
@Software: PyCharm
@modifier:
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os


"""入侵检测, 可以直接继承yolo11 重写推理方法 """
class InvasionYolo11:

    def __init__(self, ckpt_path, input_size=(640, 640), fp16=False, iou_type=None, conf_thres=0.7, iou_thres=0.4):
        self.iou_type = iou_type  # yolov5的iou_type, 这里是为了保持参数一至
        # print(conf_thres, "conf")
        self.conf_thres = conf_thres  # 这里为了检测入侵，所以默认为0.7
        self.iou_thres = iou_thres
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(ckpt_path)
        # 设置输入图像的尺寸
        self.input_size = input_size
        # 是否使用FP16精度
        self.fp16 = fp16


    def predict(self, image: np.ndarray):
        # 图像区域mask
        tensorboard_data = None
        # --------推理---------
        # 使用YOLO11模型进行预测
        results_list = self.model.predict(
            source=image,
            task='detect',
            conf=self.conf_thres,
            # conf=0.7,
            half=self.fp16,
            iou=self.iou_thres
        )
        # print(self.conf_thres)
        # 遍历检测结果
        for results in results_list:  # 这是一个ultralytics.engine.results.Results对象
            # 获取检测结果的边界框数据以及类别
            tensorboard_data = results.boxes.cpu().numpy().data
            # 打印边界框数据
            if tensorboard_data.shape[0] > 0:  # 检测到的结果大于0
                return tensorboard_data
        return tensorboard_data


if __name__ == '__main__':
    # 测试图像路径
    image = "/home/lyh/work/depoly/PyPeriShield-feature/tests/frame_0000.jpg"
    # 模型路径
    model_path = "/home/lyh/work/depoly/PyPeriShield-feature/weights/yolo11n.engine"
    # 读取图像
    img = cv2.imread(image)
    from preprocess_image import PolygonMaskProcessor
    detection_area =  [(50, 50), (200, 50), (250, 200), (50, 900), (600, 600)]
    mask = PolygonMaskProcessor(img.shape, detection_area)
    im = mask.apply_mask(img)
    # 创建PredictorYolo11实例
    predictor_yolo11 = InvasionYolo11(model_path, conf_thres=0.1)
    # 进行预测
    resource = predictor_yolo11.predict(im)
    # 打印预测结果和图像信息
    print(resource.shape[0])  # torch.Size([29, 6]) 0.5
