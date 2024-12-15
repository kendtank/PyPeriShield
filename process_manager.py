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

二期9楼摄像头的在线测试区域入侵和翻越线的demo测试

"""

import numpy as np
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
from tools import timer
import cv2
from core.tracking.tracker import BYTETracker
from core.trajectory.track_manager import TracksManager, draw_trajectories
from tools.visualization.visualize import plot_tracking
from core.detection.detector import Detector
from core.detection.invasion_region import InvasionYolo11
from services.message_queue.rabbit_mq import *
from services.video_stream.rtsp_stream import RTSPCamera
from core.detection.preprocess_image import PolygonMaskProcessor
from tools.logger import logger
from tools.load_yaml import load_yaml_file
import threading
from queue import Queue
from types import SimpleNamespace


class CameraShield:

    """ 摄像头管理类 以摄像头为单位 - 采图-推理-事件判定-回放 """

    def __init__(self, came_ip, camera_id):
        self.tracks_manage = None
        self.polygon = None   # 区域入侵mask类
        self.came_ip = came_ip
        self.arg = load_yaml_file("config/config.yaml")
        self.camera_id = camera_id
        self.camera_rtsp_url = None
        self.camera_arg = None
        self.invasion_queue = Queue(maxsize=1000)  # 解耦区域入侵算法。
        self.tracking_queue = Queue(maxsize=1000)  # 解耦目标跟踪算法。
        self.frame_id = 0  # 因为可视化视频需要。生产环境需要取消
        # 跟踪器
        self.tracker = None
        # 检测器
        self.tracking_predictor = None
        self.invasion_predictor = None
        # 初始化 MQ
        self.producer_client = ProducerClient(camera_id, broker_address="127.0.0.1", broker_port=5672)
        self.consumer_client = ConsumerClient(camera_id, broker_address="127.0.0.1", broker_port=5672)
        # 推理线程池，用来异步推理同一张图
        self.thread_pool = threading.Thread
        self.time = timer()



    # TODO 获取摄像头参数， 区域，判定线， 区域入侵区域
    def get_camera_arg(self):
        """对接redis服务器获取摄像头参数， 区域，判定线， 区域入侵区域"""
        # 检测区域， roi
        self.camera_arg = {
            "detection_area": [
                # [660, 457], [620, 1046], [1730, 1046], [1600, 322], [1100, 133]  这一快还没有实现
            ],
            # 区域入侵算法检测区域，mask
            "invasion_region": [
                [1451, 766], [1140, 1078], [1690,1079], [1810,749]
            ],
            # 判定线
            "judge_line":[
                [[981, 531], [448, 782]],
                [[920, 1078], [563, 711]]
                # [[1140, 152], [1422, 1079]]
            ]
        }

    """前期测试服务的先放到一起，后续部署在分布式集群"""
    def data_stream(self):
        """视频数据流-后续单独作为一块微服务部署在分布式集群"""
        # 初始化图像采集
        camera_stream = RTSPCamera(
            camera_id=self.camera_id,
            rtsp_url=
            "rtsp://admin:lyric1234@10.134.129.23:554/cam/realmonitor?channel=1&amp;subtype=0&amp;unicast=",
            save_dir=self.arg['cameras']['save_dir'],
            mq_producer=self.producer_client,
            fps=self.arg['cameras']['fps'],
        )
        # 采集+生产
        camera_stream.run()


    """消费图像数据 这里先放到队列解耦"""
    def get_image_path(self):
        while True:
            try:
                # --------------------消费开始-------------------------- #
                # TODO callback怎么传参数， 是不是同一个类中的消费函数和生产函数不能同时调用？
                image_path = self.consumer_client.consume_message_get()  # 使用非阻塞消费
                if image_path is not None:
                    # TODO 使用队列或线程池解耦  设置了maxsize， 会阻塞线程。
                    self.invasion_queue.put(image_path)
                    self.tracking_queue.put(image_path)
            except Exception as e:
                logger.error(f"获取图像地址错误：{e}")
                time.sleep(1)
                continue


    """区域入侵算法检测线程"""
    def process_invasion_detection_thread(self):
        while True:
            try:
                image_path = self.invasion_queue.get()  # 从队列中获取图像地址
                if image_path is not None:
                    invasion_image = cv2.imread(image_path)
                    # 区域入侵算法
                    self.invasion_detect(invasion_image)
                self.invasion_queue.task_done()  # 通知队列，任务已完成，计数器-1
            except Exception as e:
                logger.error(f"处理区域入侵算法错误：{e}")


    """区域入侵算法。后续单独作为一块微服务"""
    def invasion_detect(self, image:np.ndarray):
        # ------------------------区域入侵------------------- #  10 ms
        # 初始化检测头
        if self.invasion_predictor is None:
            self.invasion_predictor = InvasionYolo11(self.arg['algorithm_para']['ckpt_path'], conf_thres=0.8)
        if self.polygon is None:
            # 处理掩玛类
            self.polygon = PolygonMaskProcessor(image.shape, self.camera_arg["invasion_region"])
        # ------------------------
        # 获取mask
        mask = self.polygon.apply_mask(image)
        # 进行预测
        resource = self.invasion_predictor.predict(mask)
        if resource is not None:
            logger.error("人员入侵！违规！！！！!\n")
            logger.error("人员入侵！发送到事件消费类！")

    """目标追踪算法线程"""
    def process_object_tracking_thread(self):
        while True:
            try:
                image_path_ = self.tracking_queue.get()  # 从队列中获取图像地址
                if image_path_ is not None:
                    track_image = cv2.imread(image_path_)
                    # 目标追踪算法
                    self.object_tracking(track_image)
                self.tracking_queue.task_done()  # 通知队列，任务已完成，计数器-1 避免阻塞线程
            except Exception as e:
                logger.exception(f"处理目标追踪算法错误：{e}")


    """目标追踪算法流。后续单独作为一块微服务"""
    def object_tracking(self, image:np.ndarray):
        self.time.start()
        # 初始化追踪器
        if self.tracker is None:
            # 字典转为对象  SimpleNamespace 只能转一层
            tracker_arg = SimpleNamespace(**(self.arg["algorithm_para"]))
            # logger.info(f"tracker_arg:{tracker_arg}, type:{type(tracker_arg)}")
            # logger.info(f"tracker_arg2222:{tracker_arg.track_thresh}")
            # logger.info(f"self.arg:{self.arg}")
            self.tracker = BYTETracker(tracker_arg, fps=self.arg['cameras']['fps'])  # 默认最大丢失时间 2 s
        # 初始化检测头
        if self.tracking_predictor is None:
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
            # 初始化轨迹管理器
            self.tracks_manage = TracksManager(max_missed_frames=2 * self.arg["cameras"]["fps"])  # 最大丢失轨迹的时间 默认两秒
            self.tracks_manage.set_lines(self.camera_arg["judge_line"])

        # -----------------------目标追踪-------------------- #  40 ms
        # TODO 如果使用线程池：async_inference_wait --> raise RuntimeError('cannot schedule new futures after
        outputs, scale = self.tracking_predictor.inference(image)
        if outputs is not None:
            # logger.info(f"检测头推理的结果：{outputs[0]}")   # 二维张量， 每一行都是7个数
            online_targets, _ = self.tracker.update(outputs, scale)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # print("tlwh[2] / tlwh[3]", tlwh[2] / tlwh[3])
                vertical = tlwh[2] / tlwh[3] > self.arg['algorithm_para']['aspect_ratio_thresh']
                if tlwh[2] * tlwh[3] > self.arg['algorithm_para']['min_box_area']and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            """轨迹后处理"""
            # 处理当前帧的轨迹 并在这里判断是否碰线， # NOTE 规则自定义
            track_id_ = self.tracks_manage.process_frame(online_ids, online_tlwhs)
            # 获取所有轨迹id实例对象
            all_tracks = self.tracks_manage.get_all_tracks()
            # 绘制轨迹
            frame_with_trajectories = draw_trajectories(image, all_tracks,
                                                        lines=self.tracks_manage.lines,
                                                        max_length=2 * self.arg["cameras"]["fps"],  # 可视化两秒的轨迹
                                                        track_id = track_id_
                                                        )
            # 绘制检测框及id
            self.time.stop()
            online_im = plot_tracking(
                frame_with_trajectories, online_tlwhs, online_ids, frame_id=self.frame_id + 1
            )
        else:
            self.time.stop()
            online_im = image
        self.frame_id += 1
        # 显示处理后的帧
        cv2.imshow('Frame with Trajectories', online_im)
        cv2.waitKey(1)



    """摄像头主进程"""
    def start(self):
        # 获取参数
        self.get_camera_arg()
        # 启动视频流
        threading.Thread(target=self.data_stream).start()  # 启动视频流
        # 检测
        time.sleep(1)
        threading.Thread(target=self.get_image_path).start()
        threading.Thread(target=self.process_object_tracking_thread).start()
        threading.Thread(target=self.process_invasion_detection_thread).start()  # 区域入侵防线



if __name__ == "__main__":
    cs = CameraShield("webdemo001", 'webdemo001')
    cs.start()
