#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:44
@Description: demo - manage的一个demo实现
@Modify:
@Contact: tankang0722@gmail.com


# 获取摄像头参数,区域,判定线,区域入侵区域
# 采图+生产图像
# 区域入侵
# 推理+跟踪
# 轨迹后处理
# 事件判定
# 回放
# 推送
"""
import numpy as np
import os
import time
import cv2
from core.tracking.tracker import BYTETracker
from tools.visualization.visualize import plot_tracking

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(project_root)  # /home/lyh/work/depoly/PyPeriShield-feature
os.chdir(project_root)
from core.detection.detector import Detector
from core.detection.invasion_region import InvasionYolo11
from services.message_queue.rabbit_mq import RabbitMQClient
from services.video_stream.rtsp_stream import RTSPCamera
from core.detection.preprocess_image import PolygonMaskProcessor
from util.logger import logger
from util.load_yaml import load_yaml_file
import threading
from queue import Queue




class CameraShield:
    """ 摄像头管理类 以摄像头为单位 - 采图-推理-事件判定-回放 """

    def __init__(self, came_ip, camera_id):
        self.polygon = None   # 区域入侵mask
        self.came_ip = came_ip
        self.arg = load_yaml_file("config/config.yaml")
        self.camera_id = camera_id
        self.camera_rtsp_url = None
        self.camera_arg = None
        self.queue = Queue()  # 解耦两个算法。
        self.mq_client = None
        # 跟踪器
        self.tracker = None
        # self.tracker = BYTETracker(self.arg)  # 根据帧率决定缓存区
        self.frame_id = 0  # 因为可视化视频需要。生产环境需要取消
        self.tracking_predictor = None
        self.invasion_predictor = None


    # TODO 获取摄像头参数， 区域，判定线， 区域入侵区域
    def get_camera_arg(self):
        """对接redis服务器获取摄像头参数， 区域，判定线， 区域入侵区域"""
        self.camera_arg = {
            "detection_area": [(50, 50), (200, 50), (250, 200), (50, 900), (600, 600)],  # 检测区域， roi
            "invasion_region": [(5, 5), (5, 1000), (1000,1000), (1000,0)],  # 区域入侵算法检测区域，mask
            "judge_line": [(50, 50), (200, 50), (250, 200), (50, 900), (600, 600)]    # 判定线
        }


    def data_stream(self):
        # 初始化 MQ
        self.mq_client = RabbitMQClient(camera_id=self.camera_id)
        # 初始化图像采集
        camera_stream = RTSPCamera(
            camera_id=self.camera_id,
            rtsp_url=
            "rtsp://admin:lyric1234@10.134.129.23:554/cam/realmonitor?channel=1&amp;amp;subtype=0&amp;amp;unicast=",
            save_dir=self.arg['cameras']['save_dir'],
            mq_producer=self.mq_client,
            fps=self.arg['cameras']['fps'],
        )
        # 采集+生产
        camera_stream.capture_and_save_frame()


    """区域检测"""
    def invasion_detect(self, image:np.ndarray):
        # 初始化检测头
        if self.invasion_predictor is None:
            self.invasion_predictor = InvasionYolo11(self.arg['algorithm_para']['ckpt_path'], conf_thres=0.8)
        if self.polygon is None:
            self.polygon = PolygonMaskProcessor(image.shape, self.camera_arg["invasion_region"])

        mask = self.polygon.apply_mask(image)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # 进行预测
        resource = self.invasion_predictor.predict(mask)
        if resource is not None:
            print("人员入侵！违规！！！！")



    def get_image(self):
        while True:
            # print("self.my_consumer:", self.my_consumer, id(self.my_consumer))
            try:
                image_path = self.mq_client.consume_message()   # 没有就会获取空的
                if image_path is not None:
                    # logger.info("取图成功：", image_path)
                    image = cv2.imread(image_path)
                    self.invasion_detect(image)
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"获取图像地址错误：{e}")
                continue


    def run(self):
        # 获取参数
        self.get_camera_arg()
        # 启动视频流
        threading.Thread(target=self.data_stream).start()  # 启动视频流
        # mq取图 + 检测
        time.sleep(2)
        threading.Thread(target=self.get_image).start()





if __name__ == "__main__":
    cs = CameraShield("10.134.129.23", 'demo_test')
    cs.run()
