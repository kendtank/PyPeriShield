import pika
import time
from threading import Thread, Lock
import random
import string


# CameraMQ 类：每个摄像头实例拥有自己的 RabbitMQ 连接
class CameraMQ:
    def __init__(self, camera_id, broker_address="127.0.0.1", broker_port=5672):
        self.camera_id = camera_id
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.queue_name = f"camera_{camera_id}"
        self.connection = None
        self.channel = None

    def connect(self):
        """建立与 RabbitMQ 的连接并声明队列"""
        print(f"Attempting to connect for camera {self.camera_id}...")
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.broker_address, port=self.broker_port))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        print(f"Camera {self.camera_id} connected and queue {self.queue_name} declared")

    def send_message(self, message):
        """向对应的队列发送消息"""
        try:
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # 持久化消息
                )
            )
            print(f"Sent message to camera {self.camera_id}: {message}")
        except Exception as e:
            print(f"Failed to send message to camera {self.camera_id}: {e}")

    def close_connection(self):
        """关闭 RabbitMQ 连接"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            print(f"Camera {self.camera_id} connection closed")


# Consumer 类：负责从指定的摄像头队列中消费消息
class Consumer:
    def __init__(self, camera_mq: CameraMQ):
        self.camera_mq = camera_mq

    def consume_message(self):
        """消费一条消息并返回"""
        method_frame, header_frame, body = self.camera_mq.channel.basic_get(queue=self.camera_mq.queue_name,
                                                                            auto_ack=True)
        if method_frame:
            return body.decode()  # 返回解码后的消息
        else:
            print(f"No message received from camera {self.camera_mq.camera_id}.")
            return None


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
        message = consumer_instance.consume_message()
        if message:
            with message_lock:
                received_messages.append((consumer_instance.camera_mq.camera_id, message))
            print(f"Consumer for camera {consumer_instance.camera_mq.camera_id} received: {message}")
            received += 1
        else:
            print(f"Consumer for camera {consumer_instance.camera_mq.camera_id} timed out waiting for a message.")
            break



if __name__ == "__main__":
    # 创建20个摄像头实例
    cameras = [CameraMQ(i, broker_address="127.0.0.1", broker_port=5672) for i in range(1, 21)]

    # 为每个摄像头创建一个消费者实例
    consumers = [Consumer(camera) for camera in cameras]

    # 启动所有摄像头的连接
    for camera in cameras:
        camera.connect()

    # 定义每个摄像头的生产者和消费者线程
    producer_threads = []
    consumer_threads = []

    num_messages_per_camera = 10  # 每个摄像头发送的消息数量

    # 启动生产者线程
    for camera in cameras:
        producer_thread = Thread(target=producer, args=(camera, num_messages_per_camera))
        producer_threads.append(producer_thread)
        producer_thread.start()

    # 启动消费者线程
    for consumer_instance in consumers:
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

    # 关闭所有连接
    for camera in cameras:
        camera.close_connection()