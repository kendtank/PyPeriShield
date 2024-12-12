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
import logging
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
from tools import timer


def main():
    timer1 = timer()
    # 创建 Detector 实例
    detector = Detector(
        ckpt_path=r"weights/yolo11n.engine",  # 替换为实际的模型路径
        input_size=(640, 640),
        half=False,
        detector='yolo11',
        num_workers=2,
        conf_thres = 0.1,
        iou_thres = 0.4
    )

    rtsp_url = "camera001.mp4"
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        # 读取图像
        ret, frame = cap.read()
        if ret:
            timer1.start()
            # 显示图像
            detector.async_inference_wait(frame)
            print("推理一张图耗时：", timer1.stop(average=False))
            # cv2.waitKey(1)
            # cv2.imshow("RTSP", frame)
        else:
            print("采集失败！平均耗时一张图：", timer1.average_time)
            break


if __name__ == "__main__":
    time1 = time.time()
    main()
    print("time cost:", time.time() - time1)

    """
    测试视频 25 帧率 3051张图， 采用同步推理， num_workers=2
    pt 耗时： 平均耗时一张图： 5.9ms  
    onnx-gpu 耗时： cuda引擎： 平均耗时一张图： 5.87ms   tensorrt引擎： 平均耗时一张图： 4.35ms  (加载特别满，显卡资源是pt的三四倍)
    engine 耗时：平均耗时一张图： 4.45 ms
    """