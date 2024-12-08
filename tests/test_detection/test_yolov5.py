# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/8 下午1:46
@Author  : Kend
@FileName: test_yolov5.py
@Software: PyCharm
@modifier:
"""
"""
@Time    : 2024/11/23 下午11:42
@Author  : Kend
@FileName: test-.py
@Software: PyCharm
@modifier:
"""


import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)
import time
import cv2
from core.detection.detector import Detector

def main():
    # 模拟多个图像
    images = [cv2.imread("tests/frame_0000.jpg") for _ in range(100)]
    # images = [np.random.rand(640, 640, 3) for _ in range(5)]  # 生成 5 张随机图像

    # 创建 Detector 实例
    detector = Detector(
        ckpt_path=r"weights/yolov5n.pt",  # 替换为实际的模型路径
        input_size=(640, 640),
        half=False,
        detector='yolov5',  # 或 'yolov5'
        num_workers=2
    )

    # for img in images:
    #     # 执行推理任务
    #     detector.async_inference_wait(img)
    #     # detector.inference(img)
    detector.work(images)



if __name__ == "__main__":
    time1 = time.time()
    main()
    print("time cost:", time.time() - time1)

    """
    yolov5 10.35s  批量推理：8.66s
    yolo11 8.7s    批量推理：9.1s
    """
