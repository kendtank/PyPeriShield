#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/7
@Time: 09:24
@Description: test_yolo11 - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""

import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)


class PredictorYolo11:
    # 存储类的唯一实例
    _instance = None

    # 保证多个实例只有一个对象
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PredictorYolo11, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path, input_size=(640, 640), fp16=True, conf_threshold=0.1):
        if not hasattr(self, 'initialized'):
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                logging.error(f"Failed to load model from {model_path}: {e}")
                raise RuntimeError(f"Failed to load model from {model_path}: {e}")

            self.input_size = input_size
            self.fp16 = fp16
            self.conf_threshold = conf_threshold
            self.initialized = True

    def predict(self, image: np.ndarray, timer=None):
        if timer:
            timer.start()

        img_info = {
            "id": 0,
            "file_name": None,
            "height": image.shape[0],
            "width": image.shape[1],
            "raw_img": image,
            "ratio": min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
        }

        try:
            results_list = self.model.predict(
                source=image,
                task='detect',
                conf=self.conf_threshold,
                half=self.fp16
            )
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

        tensorboard_data = None
        for results in results_list:
            tensorboard_data = results.boxes.data

        return tensorboard_data, img_info


class ConcurrentPredictor:
    def __init__(self, model_path, input_size=(640, 640), fp16=True, conf_threshold=0.1, max_workers=10):
        self.predictor = PredictorYolo11(model_path, input_size, fp16, conf_threshold)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def predict(self, image: np.ndarray, timer=None):
        future = self.executor.submit(self.predictor.predict, image, timer)
        return future.result()



if __name__ == "__main__":
    model_path = "path/to/your/model.pt"
    predictor = ConcurrentPredictor(model_path)

    # 假设你有一个图像数组 image
    # image = np.array(...)
    # result = predictor.predict(image)
    # print(result)

