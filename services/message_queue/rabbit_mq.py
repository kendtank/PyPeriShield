#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:44
@Description: demo - manage的一个demo实现
@Modify:
@Contact: tankang0722@gmail.com

踩坑：
    RabbitMQ 的 Python 客户端 pika 不是线程安全的。这意味着你不能在多个线程中共享同一个 Connection 或 Channel 对象。
    如果你打算在多个线程中使用 RabbitMQ，必须确保每个线程都有自己独立的 Connection 和 Channel
"""


import pika
import threading
import time
from tools.logger import logger


class RabbitMQBase:
    def __init__(self, camera_id, broker_address="127.0.0.1", broker_port=5672):
        """根据摄像头id决定队列的声明"""
        self.camera_id = camera_id
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.queue_name = f"camera_{camera_id}"
        self.connection = None
        self.channel = None

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

    def ensure_connection(self):
        """确保连接有效，否则尝试重新连接"""
        if not self.connection or self.connection.is_closed:
            logger.info(f"Camera {self.camera_id} 尝试重新连接...")
            self.connect()

    def close_connection(self):
        """关闭 RabbitMQ 连接"""
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                logger.info(f"Camera {self.camera_id} 关闭MQ连接成功！")
        except Exception as e:
            logger.error("关闭RabbitMq连接失败了！")

    def delete_queue(self):
        try:
            self.channel.queue_delete(queue=self.queue_name)
            logger.info("队列删除成功！")
        except Exception as e:
            logger.exception("队列删除失败！")



class ProducerClient(RabbitMQBase):
    def send_message(self, message):
        """向对应的队列发送消息"""
        try:
            # 确保连接有效
            self.ensure_connection()
            if not self.channel:
                logger.error(f"Channel for camera {self.camera_id} 没有初始化！！")
                return False

            self.channel.basic_publish(
                exchange='',   # 默认使用扇出交换（Fanout Exchange）
                routing_key=self.queue_name,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # 持久化消息
                )
            )
            logger.info(f"发送数据到：camera {self.camera_id}: {message}")
            return True
        except Exception as e:
            logger.error(f"发送数据失败 camera {self.camera_id}: {e}")
            return False


class ConsumerClient(RabbitMQBase):
    def consume_message_wait(self, callback):
        """启动消费者，使用阻塞式消费"""
        try:
            self.ensure_connection()
            self.channel.basic_consume(queue=self.queue_name, on_message_callback=callback, auto_ack=True)
            logger.info(f"Camera {self.camera_id} 消费消息...")
            self.channel.start_consuming()
        except Exception as e:
            logger.error(f"摄像头消费失败：{self.camera_id}: {e}")
            raise e

    def consume_message_get(self):
        """启动消费者，使用非阻塞式消费"""
        try:
            self.ensure_connection()
            while True:
                method_frame, header_frame, body = self.channel.basic_get(
                    queue=self.queue_name, auto_ack=True)  # 非阻塞消费, 自动确认消息
                if method_frame:
                    return body.decode()
                else:
                    time.sleep(0.04)  # 一帧时间为0.04
        except Exception as e:
            logger.error(f"摄像头消费失败：{self.camera_id}: {e}")
            raise e



if __name__ == '__main__':
    pass
    """
    RabbitMQBase：这是一个基类，包含了 RabbitMQClient 的通用功能，如连接管理、队列声明等。ProducerClient 和 ConsumerClient 继承自 RabbitMQBase，分别实现了生产和消费的具体逻辑。
    ProducerClient：负责将图像路径发送到 RabbitMQ 队列中。每次发送消息时，都会调用 ensure_connection() 来确保连接有效。
    ConsumerClient：负责从 RabbitMQ 队列中获取图像路径。你可以选择使用阻塞式消费（consume_message_wait）或非阻塞式消费（consume_message_get）。在这个例子中，我们使用了非阻塞式消费，因为它不会阻塞主线程。
    CameraManager：这是主类，负责管理摄像头的图像采集和消费。它启动了两个线程：
        生产者线程：负责从 RTSP 流中捕获图像并将其保存为文件，然后将文件路径发送到 RabbitMQ 队列中。
        消费者线程：负责从 RabbitMQ 队列中获取图像路径，读取图像并进行入侵检测。
    线程安全：由于 ProducerClient 和 ConsumerClient 各自拥有独立的 Connection 和 Channel，因此它们可以在不同的线程中安全地运行，而不会相互干扰。
    资源管理：在 close() 方法中，我们确保所有的资源（如摄像头流、RabbitMQ 连接）都被正确关闭，避免资源泄露。
    """





