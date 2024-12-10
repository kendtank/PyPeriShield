#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:44
@Description: demo - manage的一个demo实现
@Modify:
@Contact: tankang0722@gmail.com
"""

import pika
import time
from threading import Thread, Lock
import random
from utils.logger import logger


# CameraMQ 类：每个摄像头实例拥有自己的 RabbitMQ 连接
class CameraMQ:
    def __init__(self, camera_id, broker_address="127.0.0.1", broker_port=5672):
        self.camera_id = camera_id
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.queue_name = f"camera_{camera_id}"
        self.connection = None
        self.channel = None
        # 实例化的同时就直接连接
        self.connect()

    def connect(self):
        """建立与 RabbitMQ 的连接并声明队列"""
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.broker_address, port=self.broker_port))
            self.channel = self.connection.channel()
            # 声明持久化队列
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            logger.info(f"Camera {self.camera_id} MQ连接成功！队列 ：{self.queue_name} 声明成功。")
            return True
        except Exception as e:
            logger.exception(f"Camera {self.camera_id} MQ连接失败！{e}")
            self.connection = None
            self.channel = None
            return False

    def send_message(self, message):
        """向对应的队列发送消息"""
        if not self.channel:
            logger.error(f"Channel for camera {self.camera_id} 没有初始化！！")
            return
        try:
            self.channel.basic_publish(
                exchange='',   # 默认使用扇出交换（Fanout Exchange）：扇出交换将消息路由到所有绑定到该交换的队列，而不考虑路由键。
                routing_key=self.queue_name,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # 持久化消息
                )
            )
            logger.info(f"发送数据到：camera {self.camera_id}: {message}")
        except Exception as e:
            logger.exception(f"发送数据失败 camera {self.camera_id}: {e}")

    def close_connection(self):
        """关闭 RabbitMQ 连接"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info(f"Camera {self.camera_id} 关闭MQ连接成功！")

    # 删除队列
    def delete_queue(self):
        self.channel.queue_delete(queue=self.queue_name)


# Consumer 类：负责从指定的摄像头队列中消费消息
class Consumer:
    def __init__(self, camera_mq: CameraMQ):
        self.camera_mq = camera_mq
        # 检查通道是否已初始化
        if not self.camera_mq.channel:
            logger.error(f"摄像头 camera {self.camera_mq.camera_id} 没有被初始化，无法消费")
            raise ValueError(f"摄像头 camera {self.camera_mq.camera_id} 没有被初始化，无法消费")

        # 设置预取计数
        try:
            self.camera_mq.channel.basic_qos(prefetch_count=1)
        except Exception as e:
            logger.exception(f"摄像头：{self.camera_mq.camera_id}MQ，设置预取计数失败：{e}")
            raise

    def consume_message(self, timeout=4):
        """消费一条消息并返回"""
        if not self.camera_mq.channel:
            logger.error(f"摄像头 camera {self.camera_mq.camera_id} 没有被初始化，无法消费")
            return None, None

        try:
            method_frame, header_frame, body = self.camera_mq.channel.basic_get(queue=self.camera_mq.queue_name,
                                                                                auto_ack=False)
            if method_frame:
                logger.info(f"从 camera {self.camera_mq.camera_id} 接收到消息: {body.decode()}")
                return body.decode(), method_frame.delivery_tag  # 返回解码后的消息和 delivery_tag
            else:
                logger.error(f"没有接受到消息{self.camera_mq.camera_id} within {timeout} 秒.")
                return None, None
        except Exception as e:
            logger.exception(f"摄像头消费失败：{self.camera_mq.camera_id}: {e}")
            return None, None

    def ack_message(self, delivery_tag):
        """手动确认消息"""
        if not self.camera_mq.channel:
            logger.error(f"摄像头 camera {self.camera_mq.camera_id} 没有被初始化，无法消费")
            return
        try:
            self.camera_mq.channel.basic_ack(delivery_tag=delivery_tag)
        except Exception as e:
            logger.exception(f"手动确认消费完成失败: {self.camera_mq.camera_id}: {e}")





# 共享的锁和消息存储
received_messages = []
message_lock = Lock()



def producer(camera_mq: CameraMQ, num_messages=10):
    """生产者线程，负责向指定的摄像头队列发送消息"""
    for i in range(num_messages):
        message = f"Message {i + 1} from camera {camera_mq.camera_id}"
        camera_mq.send_message(message)
        print(f"Producer for camera {camera_mq.camera_id} sent: {message}")
        time.sleep(random.uniform(0.1, 0.5))  # 模拟生产者的延迟


def consumer(consumer_instance: Consumer, num_messages=10):
    """消费者线程，负责从指定的摄像头队列中消费消息"""
    received = 0
    while received < num_messages:
        message, delivery_tag = consumer_instance.consume_message(timeout=10)
        if message:
            with message_lock:
                received_messages.append((consumer_instance.camera_mq.camera_id, message))
            print(f"Consumer for camera {consumer_instance.camera_mq.camera_id} received: {message}")
            try:
                # 模拟消息处理
                time.sleep(random.uniform(0.1, 0.5))  # 模拟处理延迟
                # 处理成功后手动确认消息
                consumer_instance.ack_message(delivery_tag)
                received += 1
            except Exception as e:
                print(f"Failed to process message: {e}")
                # 如果处理失败，消息会自动重新入队
        else:
            print(
                f"Consumer for camera {consumer_instance.camera_mq.camera_id} timed out waiting for a message.")
            break




if __name__ == "__main__":
    # 创建20个摄像头实例
    cameras = [CameraMQ(i, broker_address="127.0.0.1", broker_port=5672) for i in range(1, 21)]

    # 为每个摄像头创建一个消费者实例
    consumers = []

    # 启动所有摄像头的连接
    for camera in cameras:
        if not camera.connect():
            print(f"Failed to connect for camera {camera.camera_id}. Skipping this camera.")
            continue

        # 创建消费者实例并添加到列表
        try:
            consumer_instance = Consumer(camera)
            consumers.append(consumer_instance)
        except Exception as e:
            print(f"Failed to initialize consumer for camera {camera.camera_id}: {e}")
            continue

    # 定义每个摄像头的生产者和消费者线程
    producer_threads = []
    consumer_threads = []

    num_messages_per_camera = 10  # 每个摄像头发送的消息数量

    # 启动生产者线程
    for camera in cameras:
        if camera.channel:  # 确保通道已初始化
            producer_thread = Thread(target=producer, args=(camera, num_messages_per_camera))
            producer_threads.append(producer_thread)
            producer_thread.start()

    # 启动消费者线程
    for consumer_instance in consumers:
        if consumer_instance.camera_mq.channel:  # 确保通道已初始化
            consumer_thread = Thread(target=consumer, args=(consumer_instance, num_messages_per_camera))
            consumer_threads.append(consumer_thread)
            consumer_thread.start()

    # 等待所有生产者线程完成
    for thread in producer_threads:
        thread.join()

    # 等待所有消费者线程完成
    for thread in consumer_threads:
        thread.join()

    # 验证消息
    expected_messages = set()
    actual_messages = set()

    # 构建预期的消息集合
    for camera_id in range(1, 21):
        for i in range(1, num_messages_per_camera + 1):
            expected_messages.add((camera_id, f"Message {i} from camera {camera_id}"))

    # 构建实际接收到的消息集合
    for camera_id, message in received_messages:
        actual_messages.add((camera_id, message))

    # 比较预期和实际接收到的消息
    if expected_messages == actual_messages:
        print("All messages were successfully transmitted and received.")
    else:
        missing_messages = expected_messages - actual_messages
        extra_messages = actual_messages - expected_messages
        print("Message transmission failed:")
        if missing_messages:
            print(f"Missing messages: {missing_messages}")
        if extra_messages:
            print(f"Extra messages: {extra_messages}")

    # # 关闭所有连接
    # for camera in cameras:
    #     if camera.connection:  # 确保连接已建立
    #         camera.close_connection()
