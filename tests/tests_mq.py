"""
@Time    : 2024/12/6 下午8:42
@Author  : Kend
@FileName: mq.py
@Software: PyCharm
@modifier:
"""

import pika
import threading
from concurrent.futures import ThreadPoolExecutor

# 创建一个连接
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))

# 创建一个通道
channel = connection.channel()

# 声明多个队列
channel.queue_declare(queue='topic3')
channel.queue_declare(queue='topic4')


# 发送消息的函数
def send_message(queue, message):
    channel.basic_publish(exchange='', routing_key=queue, body=message)
    print(" [x] Sent %r to %r" % (message, queue))

# 接收消息的函数
def receive_message(queue):
    def callback(ch, method, properties, body):
        print(" [x] Received %r from %r" % (body, queue))
    channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


# 创建多个线程
with ThreadPoolExecutor(max_workers=3) as executor:
    executor.submit(send_message, 'topic1', 'Hello1 from topic1')
    executor.submit(send_message, 'topic2', 'Hello1 from topic2')
    executor.submit(receive_message, 'topic1')
    executor.submit(receive_message, 'topic2')
