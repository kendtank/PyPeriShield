"""
现在我们已经有了一个
RabbitMQProducer类，用于将图像文件的路径从算法检测进程发送到RabbitMQ队列。接下来，我们将设计图像存入系统，该系统负责从RabbitMQ队列中接收消息，并将SSD中的图像文件异步地复制到硬盘中。为了确保高效和可靠的处理，我们将使用
多线程池来并行处理多个写入任务，并且会实现批量处理和错误处理机制。

设计思路
单个文件写入进程：该进程负责从RabbitMQ队列中接收写入请求，并将SSD中的图像文件异步地复制到硬盘中。为了提高性能，文件写入进程内部将使用多线程池来并行处理多个写入任务。
RabbitMQ
消息队列：所有算法检测进程将通过RabbitMQ将写入请求发送给文件写入进程。RabbitMQ是一个可靠的、分布式的消息队列系统，适合处理高并发的写入请求，并且可以确保消息的持久性和可靠性。
异步I / O和批量处理：文件写入进程将使用
aiofiles和asyncio
实现非阻塞的文件复制操作，并且可以通过批量处理来减少频繁的磁盘I / O操作，提升性能。
日志记录和错误处理：为了确保系统的稳定性和可维护性，我们将添加日志记录功能，并处理可能的错误情况（如磁盘空间不足、文件复制失败等）。SSD
文件管理：由于SSD中的文件会在30分钟后被删除，文件写入进程需要定期检查SSD中的文件，并及时将它们复制到硬盘中。你可以通过RabbitMQ的消息确认机制（ACK）来确保每个文件都被成功复制。
批量处理：为了进一步提高性能，文件写入进程可以在接收到一定数量的消息后，批量处理这些消息，减少与磁盘的交互次数。
技术栈
RabbitMQ：用于实现可靠的跨进程通信。
pika：Python的RabbitMQ客户端库，用于与RabbitMQ进行交互。
aiofiles和asyncio：用于实现异步文件复制操作。
concurrent.futures.ThreadPoolExecutor：用于创建多线程池，处理多个写入任务。
shutil：用于高效地复制文件。

"""



import os
import asyncio
import aiofiles
import shutil
import pika
import concurrent.futures
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    def __init__(self, ssd_dir: str, hdd_dir: str, rabbitmq_host: str = 'localhost',
                 rabbitmq_queue: str = 'file_transfer_queue', max_workers: int = 10):
        """
        初始化RabbitMQConsumer类。

        :param ssd_dir: SSD存储的基本目录
        :param hdd_dir: 硬盘存储的基本目录
        :param rabbitmq_host: RabbitMQ服务器地址
        :param rabbitmq_queue: RabbitMQ队列名称
        :param max_workers: 线程池的最大线程数
        """
        self.ssd_dir = Path(ssd_dir)
        self.hdd_dir = Path(hdd_dir)
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_queue = rabbitmq_queue
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._ensure_directory_exists()
        self.connection = None
        self.channel = None
        self._initialize_connection()

    def _ensure_directory_exists(self):
        """确保硬盘存储目录存在"""
        if not self.hdd_dir.exists():
            self.hdd_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_connection(self):
        """初始化RabbitMQ连接和通道"""
        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.rabbitmq_host))
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.rabbitmq_queue, durable=True)  # 确保队列持久化
            logger.info(f"RabbitMQ consumer initialized for queue: {self.rabbitmq_queue}")
        except pika.exceptions.AMQPError as e:
            logger.error(f"Failed to initialize RabbitMQ connection: {e}")
            self.reconnect()

    def reconnect(self):
        """重新连接RabbitMQ"""
        logger.info("Attempting to reconnect to RabbitMQ...")
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        self._initialize_connection()

    async def copy_file(self, file_path: str):
        """
        异步复制文件从SSD到硬盘。

        :param file_path: 文件路径（相对于SSD目录）
        """
        ssd_full_path = self.ssd_dir / file_path
        hdd_full_path = self.hdd_dir / file_path

        # 确保目标目录存在
        hdd_full_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用shutil.copy2保留文件的元数据（如修改时间）
        try:
            shutil.copy2(ssd_full_path, hdd_full_path)
            logger.info(f"File copied from {ssd_full_path} to {hdd_full_path}")
        except FileNotFoundError:
            logger.error(f"File not found in SSD: {ssd_full_path}")
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")

    def process_message(self, channel, method, properties, body):
        """处理RabbitMQ消息"""
        try:
            file_path = body.decode('utf-8')  # 假设消息体是文件路径
            future = self.thread_pool.submit(asyncio.run, self.copy_file(file_path))
            future.add_done_callback(lambda f: self._acknowledge_message(channel, method.delivery_tag))
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            self._nack_message(channel, method.delivery_tag)

    def _acknowledge_message(self, channel, delivery_tag):
        """确认消息已被处理"""
        channel.basic_ack(delivery_tag=delivery_tag)

    def _nack_message(self, channel, delivery_tag):
        """拒绝消息，可以选择重新排队或丢弃消息"""
        channel.basic_nack(delivery_tag=delivery_tag, requeue=False)

    def start_consuming(self):
        """启动RabbitMQ消费者"""
        try:
            # 设置QoS，限制每个消费者最多同时处理max_workers个消息
            self.channel.basic_qos(prefetch_count=self.thread_pool._max_workers)

            # 开始消费消息
            self.channel.basic_consume(queue=self.rabbitmq_queue, on_message_callback=self.process_message)
            logger.info("Starting to consume messages from RabbitMQ...")
            self.channel.start_consuming()
        except pika.exceptions.AMQPError as e:
            logger.error(f"RabbitMQ connection error: {e}")
            self.reconnect()
            self.start_consuming()  # 重新开始消费

    def close(self):
        """关闭RabbitMQ连接"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("RabbitMQ connection closed")

    def __del__(self):
        """析构函数，确保在对象销毁时关闭连接"""
        self.close()
