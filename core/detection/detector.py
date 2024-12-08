# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:39
@Author  : Kend
@FileName: detector.py
@Software: PyCharm
@modifier:
"""

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


def singleton(cls):
    """ 单例装饰器 进程分配单例 """
    instances = {}

    def get_instance(*args, **kwargs):
        process_id = multiprocessing.current_process().pid  # Unique per-process ID
        if process_id not in instances:
            instances[process_id] = cls(*args, **kwargs)
        return instances[process_id]

    return get_instance


@singleton
class Detector:
    """检测头单例"""

    def __init__(
            self, ckpt_path, input_size=(640, 640), half=False, num_workers=1, detector='yolo11'):
        self.detector = detector
        self.input_size = input_size
        self.ckpt_path = ckpt_path
        self.half = half
        self.num_workers = num_workers
        self.model = None
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)  # 线程池提供异步
        # NOTE 发现这种场景下，线程池的管理开销超过了多线程带来的并行加速效果。
        # 加载检测头
        self.get_detector()
        self._shutdown = False


    def get_detector(self):
        if self.model is None:
            with self.lock:
                if self.model is None:  # 确保多进程的线程安全 -> 单例模式
                    if self.detector == 'yolo11':
                        from core.detection.yolo11.predictor_yolo11 import PredictorYolo11
                        self.model = PredictorYolo11(
                            self.ckpt_path, self.input_size, fp16=self.half
                        )
                    elif self.detector == 'yolov5':
                        from core.detection.yolov5.predictor_yolov5 import Yolov5Predictor
                        self.model = Yolov5Predictor(self.ckpt_path, self.input_size, self.half, iou_type='giou')
                    else:
                        # NOTE 留给yolo12
                        raise ValueError(f"Unknown detector: {self.detector}")

    # 瓶颈在于计算，而不在于并发量。所以没有改为异步函数了 。 异步特性主要用于 I/O 密集型任务
    def inference(self, image: np.ndarray):
        # with self.lock:  # NOTE 海葵云的分布式集群验证不需要锁，所以这里注释掉
        # 新加的检测头需要定义推理为predict函数名
        tensor_board_data, ratio = self.model.predict(image)
        return tensor_board_data, ratio

    # BUG
    def async_inference(self, image: np.ndarray):
        """提交一个请求到线程池做异步"""
        return self.executor.submit(self.inference, image)

    def async_inference_wait(self, image: np.ndarray):
        """提交一个请求到线程池做异步，并同步等待结果"""
        future = self.executor.submit(self.inference, image)
        try:
            result = future.result()  # 等待任务完成并获取结果
            return result
        except Exception as e:
            print(f"推理过程中发生错误: {e}")
            return None

    def shutdown(self, wait=True):
        """关闭线程池"""
        self._shutdown = True
        self.executor.shutdown(wait=wait)


    # 批处理方式才会提升fps， 但是场景用不到，多线程已经足够快了 # NOTE 提升在5%
    def work(self, images: list[np.ndarray]):
        """
        Args:
            images: List of images to process.
        Returns:
            List of inference results.
        """
        tasks = [self.async_inference(image) for image in images]
        # 等待所有任务完成返回结果
        results = []
        for future in as_completed(tasks):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error during inference: {e}")
        return results
