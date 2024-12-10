"""
@Time    : 2024/12/6 下午8:42
@Author  : Kend
@FileName: mq.py
@Software: PyCharm
@modifier:
"""
import time
from threading import Thread, Lock
import random
import string
from services.message_queue.mq import Consumer, CameraMQ


# CameraMQ 和 Consumer 类保持不变，使用没有线程池的版本


import pika
from contextlib import contextmanager
import time
from pika.exceptions import AMQPConnectionError, ChannelClosedByBroker, UnroutableError
from threading import Thread, Lock
import random
import string

# CameraMQ 和 Consumer 类保持不变

# 共享的锁和消息存储
received_messages = []
message_lock = Lock()


def producer(camera_mq: CameraMQ, num_messages=10):
    """生产者线程，负责向指定的摄像头队列发送消息"""
    for i in range(num_messages):
        message = f"Message {i + 1} from camera {camera_mq.camera_id}"
        try:
            camera_mq.send_message(message)
            print(f"Producer for camera {camera_mq.camera_id} sent: {message}")
        except (AMQPConnectionError, ChannelClosedByBroker, UnroutableError) as e:
            print(f"Producer for camera {camera_mq.camera_id} failed to send message: {e}")
            # 尝试重新建立连接并再次发送消息
            if not camera_mq.connection or camera_mq.connection.is_closed:
                camera_mq._connect()
                camera_mq.send_message(message)
        time.sleep(random.uniform(0.1, 0.5))  # 模拟生产者的延迟


def consumer(consumer_instance: Consumer, num_messages=10):
    """消费者线程，负责从指定的摄像头队列中消费消息"""
    received = 0
    while received < num_messages:
        try:
            message = consumer_instance.consume_message(timeout=10)
            if message:
                with message_lock:
                    received_messages.append((consumer_instance.camera_mq.camera_id, message))
                print(f"Consumer for camera {consumer_instance.camera_mq.camera_id} received: {message}")
                received += 1
            else:
                print(f"Consumer for camera {consumer_instance.camera_mq.camera_id} timed out waiting for a message.")
                break
        except (AMQPConnectionError, ChannelClosedByBroker, UnroutableError) as e:
            print(f"Consumer for camera {consumer_instance.camera_mq.camera_id} failed to receive message: {e}")
            # 尝试重新建立连接并继续消费
            if not consumer_instance.camera_mq.connection or consumer_instance.camera_mq.connection.is_closed:
                consumer_instance.camera_mq._connect()
        except Exception as e:
            print(f"Unexpected error in consumer for camera {consumer_instance.camera_mq.camera_id}: {e}")
            break


def generate_random_string(length=10):
    """生成随机字符串，用于模拟不同的消息内容"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


if __name__ == "__main__":
    # 创建20个摄像头实例
    cameras = [CameraMQ(i, broker_address="127.0.0.1", broker_port=5672) for i in range(1, 2)]

    # 为每个摄像头创建一个消费者实例
    consumers = [Consumer(camera, max_retries=-1, retry_delay=5) for camera in cameras]

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