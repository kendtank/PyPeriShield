# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/13
@Time: 14:44
@Description:
@Modify:
@Contact: tankang0722@gmail.com

每个算法检测进程都可以实例化该类，并使用它来发送事件消息。
通过这种方式，你可以更方便地管理RabbitMQ连接和通道的生命周期，并且可以在类中添加更多的功能（如批量发送、错误处理等）。

设计思路
    EventRabbitMQProducer 类：
        我们将创建一个 EventRabbitMQProducer 类，负责初始化RabbitMQ连接和通道，并提供发送消息的功能。该类将在每个算法检测进程中实例化一次，并保持连接和通道的持久化。
    单例模式：
        虽然每个进程会实例化一个 EventRabbitMQProducer 对象，但每个对象内部的RabbitMQ连接和通道将保持持久化，避免频繁地创建和销毁资源。
        批量发送：为了提高性能，EventRabbitMQProducer 类可以实现批量发送消息的功能，减少与RabbitMQ的交互次数。
        错误处理和重连机制：为了确保系统的健壮性，EventRabbitMQProducer 类将包含错误处理和自动重连机制。如果RabbitMQ连接中断，生产者将尝试重新连接，并继续发送未发送的消息。
        日志记录：我们将在类中添加日志记录功能，方便调试和监控。

"""

import pika
import logging
from functools import lru_cache
import time
from typing import List
from tools.logger import logger



class EventRabbitMQProducer:
    def __init__(self, rabbitmq_host: str = 'localhost', rabbitmq_queue: str = 'file_transfer_queue'):
        """
        初始化RabbitMQProducer类。
        :param rabbitmq_host: RabbitMQ服务器地址
        :param rabbitmq_queue: RabbitMQ队列名称
        """
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_queue = rabbitmq_queue
        self.connection = None
        self.channel = None
        self._initialize_connection()

    def _initialize_connection(self):
        """初始化RabbitMQ连接和通道"""
        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.rabbitmq_host))
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.rabbitmq_queue, durable=True)  # 确保队列持久化
            logger.info(f"RabbitMQ producer initialized for queue: {self.rabbitmq_queue}")
        except Exception as e:
            logger.error(f"Failed to initialize RabbitMQ connection: {e}")
            self.reconnect()

    def reconnect(self):
        """重新连接RabbitMQ"""
        logger.info("Attempting to reconnect to RabbitMQ...")
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        self._initialize_connection()

    def send_message(self, file_path: str):
        """
        发送事件结果到RabbitMQ队列。
        :param file_path: 文件路径 - 具体看事件的类型修改
        """
        try:
            self.channel.basic_publish(
                exchange='',
                routing_key=self.rabbitmq_queue,
                body=file_path,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # 确保消息持久化
                )
            )
            logger.info(f"Sent message to RabbitMQ: {file_path}")
        except Exception as e:
            logger.error(f"RabbitMQ connection error: {e}")
            self.reconnect()
            self.send_message(file_path)  # 重新发送消息

    def send_messages_batch(self, file_paths: List[str]):
        """
        批量发送文件路径到RabbitMQ队列。

        :param file_paths: 文件路径列表
        """
        try:
            for file_path in file_paths:
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self.rabbitmq_queue,
                    body=file_path,
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # 确保消息持久化
                    )
                )
                logger.info(f"Sent message to RabbitMQ: {file_path}")
        except Exception as e:
            logger.error(f"RabbitMQ connection error: {e}")
            self.reconnect()
            self.send_messages_batch(file_paths)  # 重新发送消息

    def close(self):
        """关闭RabbitMQ连接"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("RabbitMQ connection closed")

    def __del__(self):
        """析构函数，确保在对象销毁时关闭连接"""
        self.close()

