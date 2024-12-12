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

import os
import time
import cv2
from core.tracking.tracker import BYTETracker
from util.visualization.visualize import plot_tracking

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(project_root)  # /home/lyh/work/depoly/PyPeriShield-feature
os.chdir(project_root)
from core.detection.detector import Detector
from core.detection.invasion_region import InvasionYolo11
from services.message_queue.rabbit_mq import CameraMQ, Consumer
from services.video_stream.rtsp_stream import RTSPCamera
from core.detection.preprocess_image import PolygonMaskProcessor
from util.logger import logger
from util.load_yaml import load_yaml_file
import threading
from queue import Queue


# self.arg = load_yaml_file("config/config.yaml")
# t = tuple(self.arg['algorithm_para']['input_size'])
# print(t, type(t))



class CameraShield:
    """ 摄像头管理类 以摄像头为单位 - 采图-推理-事件判定-回放 """

    def __init__(self, came_ip, camera_id):
        self.polygon = None   # 区域入侵mask
        self.came_ip = came_ip
        self.arg = load_yaml_file("config/config.yaml")
        self.camera_id = camera_id
        self.camera_rtsp_url = None
        self.camera_arg = None
        self.my_consumer = None
        self.queue = Queue()  # 解耦两个算法。
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
            "invasion_region": [(50, 50), (200, 50), (250, 200), (50, 900), (600, 600)],  # 区域入侵算法检测区域，mask
            "judge_line": [(50, 50), (200, 50), (250, 200), (50, 900), (600, 600)]    # 判定线
        }
        self.polygon = PolygonMaskProcessor((1080, 1920), self.camera_arg["invasion_region"])

    def data_stream(self):
        # TODO 获取摄像头参数， 区域，判定线， 区域入侵区域
        # time.sleep(5)  # 等一下模型加载
        # self.get_camera_arg()
        # 初始化配置参数
        # 初始化 MQ 生产者
        mq_producer = CameraMQ(camera_id=self.camera_id)
        self.my_consumer = Consumer(mq_producer)
        # 初始化图像采集
        camera_stream = RTSPCamera(
            camera_id=self.camera_id,
            rtsp_url="camera002.mp4",
            save_dir=self.arg['cameras']['save_dir'],
            mq_producer=mq_producer
        )
        # 采集+生产
        camera_stream.capture_and_save_frame()
        # 建立一个消费者  NOTE : 这个时候视频流是死循环
        # self.my_consumer = Consumer(mq_producer)
        # print("消费者建立成功！！")
        # print("self.my_consumer_____:", self.my_consumer, id(self.my_consumer))

    """区域检测"""
    def invasion_detect(self, image):
        # 初始化检测头
        if self.invasion_predictor is None:
            self.invasion_predictor = InvasionYolo11(self.arg['algorithm_para']['ckpt_path'], conf_thres=0.8)

        mask = self.polygon.apply_mask(image)
        # 进行预测
        resource = self.invasion_predictor.predict(mask)
        if resource is not None:
            print("人员入侵！违规！！！！")


    """目标跟踪"""
    def track_detect(self, image):
        if self.tracking_predictor is None:
            self.tracker = BYTETracker(self.arg)
            self.tracking_predictor = Detector(
                self.arg['algorithm_para']['ckpt_path'],
                self.arg['algorithm_para']['confidence_threshold'],
                self.arg['algorithm_para']['iou_threshold'],
                tuple(self.arg['algorithm_para']['input_size']),
                self.arg['algorithm_para']['half'],
                self.arg['algorithm_para']['iou_type'],
                self.arg['algorithm_para']['num_workers'],
                self.arg['algorithm_para']['model_type'],
            )
        # 推理+跟踪
        outputs, ratio = self.tracking_predictor.inference(image)  # 默认使用yolo11
        if outputs is not None:
            online_targets = self.tracker.update(outputs, ratio)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.arg['algorithm_para']['[aspect_ratio_thresh']
                if tlwh[2] * tlwh[3] > self.arg['algorithm_para']['min_box_area'] and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            online_im = plot_tracking(
                image, online_tlwhs, online_ids, frame_id=self.frame_id + 1)
        else:
            online_im = image
        cv2.imshow("ai", online_im)
        cv2.waitKey(1)
        self.frame_id += 1


    def get_image(self):
        while True:
            # print("self.my_consumer:", self.my_consumer, id(self.my_consumer))
            try:
                image_path, tags = self.my_consumer.consume_message()   # 没有就会获取空的
                if image_path is not None:
                    image = cv2.imread(image_path)
                    self.queue.put(image)
            except:
                continue
            # self.my_consumer.ack_message(tags)  # 代表接受消息完成， mq会移除消息  TODO 堵舍摸索

            # logger.info(f"消费者解码结果：{q, w}")


    def run(self):
        # 获取参数
        self.get_camera_arg()
        # 启动视频流
        threading.Thread(target=self.data_stream).start()  # 启动视频流
        # 启动取图
        # time.sleep(1)
        # mq取图
        threading.Thread(target=self.get_image).start()
        # 检测
        threading.Thread(target=self.detect).start()


    def detect(self):
        while True:
            image = self.queue.get()
            # 区域入侵检测：
            # threading.Thread(target=self.invasion_detect, args=(image,)).start()
            self.invasion_detect(image)
            # 轨迹跟踪
            self.track_detect(image)
        # 轨迹后处理
        # 事件判定
        # 回放
        # 推送


if __name__ == "__main__":
    cs = CameraShield("demo001", 'demo001')
    cs.run()
