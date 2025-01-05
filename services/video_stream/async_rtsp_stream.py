#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:44
@Description: 实现异步的rtsp采图 待完成--
@Modify:
@Contact: tankang0722@gmail.com
"""
# NOTE: 可以考虑存图和发送mq做异步处理 - 待优化项 (取消，mq速度比存图快很多)

import cv2
import os
import time
from datetime import datetime
from tools.logger import logger
from threading import Thread, Lock
from services.message_queue.rabbit_mq import *



class RTSPCamera:

    def __init__(self, camera_id, rtsp_url, save_dir, mq_producer, fps=25):
        """
        初始化 RTSPCamera 类
        :param camera_id: 摄像头 ID
        :param rtsp_url: RTSP 流地址
        :param save_dir: 图片保存目录
        :param mq_producer: 已封装好的 MQ 生产者实例
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.save_dir = save_dir
        self.mq_producer = mq_producer
        self.fps = fps  # 抽帧
        self.ori_fps = None
        # 共享锁，确保多线程环境下对 MQ 的安全访问 这里其实用不着
        self.mq_lock = Lock()

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # 打开 RTSP 流
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            logger.exception(f"RTSP视频流打开失败：{self.camera_id}.")
            raise Exception(f"RTSP视频流打开失败：{self.camera_id}.")
        # 获取原始帧率
        self.ori_fps = self.cap.get(cv2.CAP_PROP_FPS)


    def capture_and_save_frame(self):
        """从 RTSP 流中持续捕获帧并保存为图片"""
        counts = 0
        while True:
            try:
                ret, frame = self.cap.read()
                counts += 1

                if not ret:
                    logger.error(f"RTSP采集图像失败：{self.camera_id}.")
                    time.sleep(1)  # 等待1秒后重试
                    continue

                # 抽帧
                # counts = 1  2  3  。。。10  fps = 10
                if counts % self.fps != 0:
                    continue
                counts = 0   # 保证计时器一直在 0 - 10
                # 创建保存目录
                now_time = datetime.now().strftime("%Y%m%d%H%M")
                base_save_dir = f"{self.save_dir}/{now_time}"
                # 确保保存目录存在
                os.makedirs(base_save_dir, exist_ok=True)
                # 获取当前时间戳（精确到微秒）1/25 毫秒足以, 这里使用微秒
                current_time = int(time.time() * (10 ** 6))
                filename = os.path.join(base_save_dir, f"{current_time}_{self.camera_id}.jpg")

                # 保存图片
                try:
                    cv2.imwrite(filename, frame)
                    # logger.info(f"保存图像成功： {self.camera_id} to {filename}")
                except Exception as e:
                    logger.error(f"保存图像失败：{self.camera_id}: {e}")
                    continue

                # 将图片路径发送到 MQ
                self.send_message_to_mq(filename)

            except Exception as e:
                logger.error(f"Error in capture_and_save_frame for camera {self.camera_id}: {e}")
                time.sleep(1)  # 等待1秒后重试


    def send_message_to_mq(self, message):
        """将图片路径发送到 MQ"""
        with self.mq_lock:
            try:
                self.mq_producer.send_message(message)
                # logger.info(f"发送图像路径到MQ: {message}")
            except Exception as e:
                logger.error(f"发送图像路径到MQ失败: {e}")


    def run(self):
        """启动 RTSP 流捕获和图片保存"""
        logger.info(f"开启RTSP视频流采集：{self.camera_id}...")
        self.capture_and_save_frame()


    def stop(self):
        """关闭 RTSP 流"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info(f"Closed RTSP stream for camera {self.camera_id}.")





if __name__ == "__main__":
    # 配置摄像头信息
    cameras = [
        {"camera_id": 'demo001', "rtsp_url": "/home/lyh/work/depoly/PyPeriShield-feature/camera001.mp4"},
        {"camera_id": 'demo002', "rtsp_url": "/home/lyh/work/depoly/PyPeriShield-feature/camera002.mp4"},
    ]

    # 初始化 MQ 生产者
    # mq_producer = CameraMQ(camera_id="demo001")
    # logger.info(f"MQ生产者初始化成功")
    # logger.info(f"MQ生产者初始化成功：{id(mq_producer)}")
    # logger.info(f"MQ生产者初始化成功：{mq_producer.camera_id}, 'queue_name':{mq_producer.queue_name}")
    # # mq_producer.send_message("hello world")
    # # time.sleep(1)
    # # 初始化消费者
    # my_consumer = Consumer(mq_producer)
    # logger.info(f"MQ消费者初始化成功：{id(my_consumer)}")
    # for i in range(10000):
    #     py_db, tag = my_consumer.consume_message()
    #     # 处理成功后手动确认消息
    #     my_consumer.ack_message(tag)
    #     logger.info(f"MQ消费者接受消息成功:{py_db}")


    # 启动每个摄像头的 RTSPCamera 实例
    camera_instances = []
    threads = []
    save = "/home/lyh/temp_images"
    for camera in cameras:
        # 初始化 MQ 生产者
        mq_producer = CameraMQ(camera_id=camera["camera_id"])
        # logger.info(f"MQ生产者初始化成功：{id(mq_producer)}")
        camera_instance = RTSPCamera(
            camera_id=camera["camera_id"],
            rtsp_url=camera["rtsp_url"],
            save_dir=save,
            mq_producer=mq_producer
        )
        camera_instances.append(camera_instance)
        # 启动线程
        thread = Thread(target=camera_instance.run)
        threads.append(thread)
        thread.start()

    # 等待所有线程完成（实际上这些线程会一直运行）
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Stopping all cameras...")
        for camera_instance in camera_instances:
            camera_instance.stop()
        # mq_producer.close_connection()
        logger.info("All cameras stopped and MQ connection closed.")