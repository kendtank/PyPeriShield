import time
import pika
from contextlib import contextmanager
import threading
from pika.exceptions import AMQPConnectionError

"""
在 RabbitMQ 中，主题交换（Topic Exchange）和队列（Queue）是两个重要的概念。
主题交换（Topic Exchange）：
    主题交换允许您根据消息的路由键将消息发送到多个队列。
    路由键是一个字符串，可以包含一个或多个单词，单词之间用点（.）分隔。
    主题交换的路由规则允许您使用通配符（*）和（#）来匹配路由键。
    例如，路由键 camera1.image 可以匹配 camera1.* 和 camera1.#。

队列（Queue）：队列是消息的容器。
    生产者将消息发送到交换，交换根据路由规则将消息路由到一个或多个队列。
    消费者从队列中接收消息。
"""


class CameraMQ:
    """
    每个 CameraMQ 实例都会创建自己的 RabbitMQ 连接。
    因为在 RabbitMQ 中，每个连接都是独立的，每个连接都有自己的通道。
    每个通道都有自己的队列和消费者。
    NOTE 也可以使用连接池来管理连接。连接池可以重用连接，而不是为每个 Camera 实例创建新的连接。我的生存环境不用频繁建立连接
    """
    def __init__(self, camera_id, broker_address="127.0.0.1", broker_port=5672):
        self.camera_id = camera_id
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.queue_name = f"camera_{camera_id}"
        self.connection = None
        self.channel = None


    def _connect(self):
        """建立与 RabbitMQ 的连接并声明队列"""
        print(f"建立MQ连接：{self.camera_id}...")
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.broker_address, port=self.broker_port))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        print(f"Camera {self.camera_id} 连接成功 and 队列 {self.queue_name} 声明成功")


    def send_message(self, message):
        """向对应的队列发送消息"""
        if not self.connection or self.connection.is_closed:
            self._connect()
        try:
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=message,
                # properties=pika.BasicProperties(
                #     delivery_mode=2,  # 持久化消息
                # )
            )
            print(f"生产消息：{self.camera_id}: {message}")
        except AMQPConnectionError:
            print(f"连接失败：{self.camera_id}. Retrying...")
        #     self._connect()  # 重新建立连接后再次尝试发送
            # self.send_message(message)  # 递归调用自身以确保消息发送成功


    @contextmanager
    def connect(self, max_retries=5, retry_delay=5):
        """上下文管理器用于自动管理连接"""
        retries = 0
        while retries < max_retries or max_retries == -1:  # -1 表示无限重试
            try:
                if not self.connection or self.connection.is_closed:
                    self._connect()
                yield self.channel
                break  # 如果连接成功，退出循环
            except AMQPConnectionError as e:
                print(f"MQ连接失败: {e}. Retrying in {retry_delay} seconds...")
                retries += 1
                time.sleep(retry_delay)
            except Exception as e:
                print(f"mq异常 error: {e}. Stopping consumer.")
                break
        else:
            print(f"超出最大重连次数： {self.camera_id}.")
            raise ConnectionError("MQ超出最大重连次数")

        self.close_connection()


    def close_connection(self):
        """关闭 RabbitMQ 连接"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            print(f"Camera {self.camera_id} 关闭连接...")



"""消费"""
class Consumer:
    def __init__(self, camera_mq: CameraMQ, max_retries=5, retry_delay=5):
        self.camera_mq = camera_mq
        self.max_retries = max_retries
        self.retry_delay = retry_delay


    def consume_message(self, timeout=None):
        """消费一条消息并返回"""
        with self.camera_mq.connect(max_retries=self.max_retries, retry_delay=self.retry_delay) as channel:
            method_frame, header_frame, body = channel.basic_get(queue=self.camera_mq.queue_name, auto_ack=True)
            if method_frame:
                # if body:
                #     camera_message = body.decode()  # 解码后的消息
                print(f"接受到消息：{self.camera_mq.camera_id}")
                return body.decode()  # 返回解码后的消息
            else:
                print(f"没有消息接受：{self.camera_mq.camera_id}.")
                return None



# """消费者, 回调手动确认消息处理完"""
# class Consumer:
#     def __init__(self, camera_mq: CameraMQ, max_retries=5, retry_delay=5):
#         self.camera_mq = camera_mq
#         self._running = False
#         self.max_retries = max_retries  # 最大重试次数  -1 表示无限重试
#         self.retry_delay = retry_delay  # 重试间隔时间（秒）
#
#
#     def start_consuming(self):
#         """开始监听队列中的消息，并实现自动重连"""
#         retries = 0  # 重试次数
#         # 用于在断开时自动尝试重连保证数据可靠性
#         while retries < self.max_retries or self.max_retries == -1:  # -1 表示无限重试
#             try:
#                 # 拿到生产者的连接, 并判断连接正常
#                 with self.camera_mq.connect() as channel:
#                     def callback(ch, method, properties, body):
#                         print(f"Received message from camera {self.camera_mq.camera_id}: {body.decode()}")
#                         # 在这里处理接收到的消息
#                         self.process_message(body)
#                         # 如果需要确认消息已处理，可以在这里调用channel.basic_ack(delivery_tag=method.delivery_tag)
#
#                     channel.basic_consume(queue=self.camera_mq.queue_name, on_message_callback=callback, auto_ack=True)
#                     print(f"[*] Camera {self.camera_mq.camera_id} waiting for messages. To exit press CTRL+C")
#                     self._running = True
#                     try:
#                         channel.start_consuming()
#                     except KeyboardInterrupt:
#                         print(f"Interrupted by user: Camera {self.camera_mq.camera_id}")
#                         self.stop_consuming()
#                         return  # 退出循环
#             # 捕获连接错误并等待后重试
#             except AMQPConnectionError as e:
#                 print(f"Connection error: {e}. Retrying in {self.retry_delay} seconds...")
#                 retries += 1
#                 time.sleep(self.retry_delay)
#             # 捕获其他异常并停止消费者
#             except Exception as e:
#                 print(f"Unexpected error: {e}. Stopping consumer.")
#                 self.stop_consuming()
#                 break
#
#         if retries >= self.max_retries and self.max_retries != -1:
#             print(f"Max retries reached. Stopping consumer for camera {self.camera_mq.camera_id}.")
#             self.stop_consuming()
#
#
#     def stop_consuming(self):
#         """停止监听"""
#         if self._running:
#             self._running = False
#             self.camera_mq.close_connection()
#
#     # 处理图像
#     def process_message(self, body):
#         """处理接收到的消息"""
#         # 在这里实现具体的业务逻辑
#         print(f"Processing message: {body.decode()}")




# 示例用法
if __name__ == "__main__":
    cameras = [CameraMQ(i) for i in range(1)]  # 创建20个摄像头实例
    # 使用线程来并发启动所有生产者
    threads1 = []
    for camera in cameras:
        thread = threading.Thread(target=camera.send_message, args=(f"Hello, {camera.camera_id}", ))
        threads1.append(thread)
        thread.start()

    consumers = [Consumer(camera) for camera in cameras]  # 为每个摄像头创建一个消费者实例
    # 使用线程来并发启动所有消费者的监听
    threads2 = []
    for consumer in consumers:
        thread = threading.Thread(target=consumer.consume_message)
        threads2.append(thread)
        thread.start()


    # 等待所有线程完成（在实际应用中，这可能永远不会发生，除非有特定的退出条件）
    for thread in threads2:
        thread.join()

