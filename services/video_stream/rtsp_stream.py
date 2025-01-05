#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:44
@Description: rtsp的一个实现， 之前封装过三种方式
1: 一直获取帧号并解码（根据需要解码帧）cap.grab() 只抓取下一帧数据，而不解码。
    说明：这种方法用于持续获取视频流帧号，但仅在需要时解码帧数据。通过 cv2.VideoCapture 的 CAP_PROP_POS_FRAMES 属性实时获取当前帧号。
2：采集需要的帧后休眠，必要时跳到最新帧号
    说明：通过调用 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number) 将读取位置跳到最新需要的帧号。当没有新需求时，可以通过休眠降低资源消耗。
    cap.set() 允许将视频流的位置设置为指定帧号。
1：优化项：
    创建一个缓冲区：用于存储最新的帧和帧号。
    后台线程：持续从 RTSP 流中读取帧，并更新缓冲区中的帧号。
    主程序：在需要时，从缓冲区中获取最新的帧并进行解码。
2：优化项：
    创建一个控制标志：用于控制是否继续采集帧。
    后台线程：根据控制标志决定是否set后继续采集帧。
    主程序：在需要时，设置控制标志为 True 并重新开始采集，不需时设置为 False 并暂停采集。
方法1: 12th i7 千兆的主机可以支持40个视频流的采集， 方法2：能够支持80组，但是对延时特别敏感。建议不要使用
@Modify:
@Contact: tankang0722@gmail.com
"""

import cv2
import os
import time
from datetime import datetime
from services.message_queue.rabbit_mq import ProducerClient
from tools.logger import logger
from threading import Thread, Lock


class RTSPCamera:

    def __init__(self, camera_id, rtsp_url, save_dir, mq_producer:ProducerClient, fps=25):
        """
        初始化 RTSPCamera 类
        :param camera_id: 摄像头 ID
        :param rtsp_url: RTSP 流地址
        :param save_dir: 图片保存目录
        :param mq_producer: 已封装好的 MQ 生产者实例
        """
        self.cap = None
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.save_dir = save_dir
        self.mq_producer = mq_producer
        self.fps = fps  # 抽帧, 默认1s抽一张
        self.ori_fps = None
        # 共享锁，确保多线程环境下对 MQ 的安全访问
        self.mq_lock = Lock()
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        self.lost_time = 0


    def capture_and_save_frame(self):
        """从 RTSP 流中持续捕获帧并保存为图片, 传递到MQ中"""
        while True:
            self.cap = None
            time.sleep(2)
            # 打开 RTSP 流
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                logger.exception(f"RTSP视频流打开失败：{self.camera_id}.")
                time.sleep(5)
                self.cap.release()
            # 获取原始帧率
            self.ori_fps = self.cap.get(cv2.CAP_PROP_FPS)
            counts = 0

            # 采集图像
            while True:
                try:
                    ret, frame = self.cap.read()
                    counts += 1
                    if not ret:
                        logger.error(f"RTSP采集图像失败：{self.camera_id}.")
                        time.sleep(1)  # 等待1秒后重试
                        self.lost_time += 1
                        if self.lost_time >= 5:
                            self.cap.release()
                            break
                        continue

                    # 抽帧
                    # counts = 1  2  3  。。。10  fps = 10
                    if counts % self.fps != 0:
                        time.sleep(0.01)  # 生产环境禁止使用
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
                        filename = self.save_image(frame, filename)
                        # 将图片路径发送到 MQ
                        self.send_message_to_mq(filename)
                    except Exception as e:
                        logger.error(f"保存图像失败：{self.camera_id}: {e}")
                        continue
                except Exception as e:
                    logger.error(f"Error in capture_and_save_frame for camera {self.camera_id}: {e}")
                    time.sleep(2)  # 等待2秒后重试
                    self.cap.release()
                    break


    def save_image(self, frame, filename):
        # 保存图片
        # 保存图片
        cv2.imwrite(filename, frame)
        return filename

    def send_message_to_mq(self, message):
        """将图片路径发送到 MQ"""
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





"""使用opencv的grab 和 retrieve 减少cpu使用率 """
class RTSPClientGrab:
    def __init__(self, rtsp_url, target_fps=5, source_fps=25, reconnect_delay=5):
        """
        :param rtsp_url: RTSP 流地址
        :param target_fps: 需要采集的帧率（如 5）
        :param source_fps: 原始视频流的帧率（如 25）
        :param reconnect_delay: 断线重连的时间间隔（单位：秒）
        """
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.frame_interval = int(source_fps / target_fps)  # 每隔多少帧采一次
        self.reconnect_delay = reconnect_delay
        self.running = False
        self.capture = None
        self.last_success_time = None  # 记录最后一次成功采集帧的时间

    def connect(self):
        """尝试连接到 RTSP 流"""
        if self.capture:
            self.capture.release()

        self.capture = cv2.VideoCapture(self.rtsp_url)
        if not self.capture.isOpened():
            print(f"[ERROR] Failed to connect to {self.rtsp_url}")
            return False

        print(f"[INFO] Successfully connected to {self.rtsp_url}")
        self.last_success_time = time.time()  # 更新成功连接时间
        return True

    def check_connection(self):
        """检查是否需要重连"""
        if not self.capture or not self.capture.isOpened():
            print("[WARNING] Connection lost, attempting to reconnect...")
            self.connect()
            time.sleep(self.reconnect_delay)

    def read_and_process(self):
        """读取和处理帧"""
        while self.running:
            self.check_connection()  # 检查连接状态

            for _ in range(self.frame_interval - 1):
                if not self.capture.grab():  # 尝试跳帧
                    print("[WARNING] Failed to grab frame, reconnecting...")
                    self.check_connection()
                    break

            # 获取目标帧
            ret, frame = self.capture.retrieve()
            if not ret:
                print("[WARNING] Failed to retrieve frame, reconnecting...")
                self.check_connection()
                continue

            self.last_success_time = time.time()  # 更新成功处理帧的时间
            self.process_frame(frame)

            # 保证实时性
            time.sleep(1 / self.target_fps)

    def process_frame(self, frame):
        """对帧进行处理，这里仅示例显示"""
        print("[INFO] Processing frame")
        # 示例：仅显示帧（实际可替换为保存帧或推送到其他服务）
        # cv2.imshow("Frame", frame)
        # cv2.waitKey(1)

    def start(self):
        """启动 RTSP 流采集"""
        self.running = True
        if not self.connect():
            print("[ERROR] Initial connection failed, exiting...")
            return

        try:
            self.read_and_process()
        except Exception as e:
            print(f"[ERROR] Exception occurred: {e}")
        finally:
            self.stop()

    def stop(self):
        """停止 RTSP 流采集"""
        self.running = False
        if self.capture:
            self.capture.release()
        print("[INFO] Stream stopped")


# # 使用示例
# if __name__ == "__main__":
#     rtsp_url = "rtsp://username:password@ip_address:port/stream"
#     client = RTSPClient(rtsp_url, target_fps=5, source_fps=25)
#
#     try:
#         client.start()
#     except KeyboardInterrupt:
#         client.stop()

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
        mq_producer = ProducerClient(camera_id=camera["camera_id"])
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