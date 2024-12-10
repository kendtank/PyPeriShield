import pika
from contextlib import contextmanager
from threading import Lock
import time
from pika.exceptions import AMQPConnectionError


"""连接池应该具备重连机制，保证高可用性"""
class ConnectionPool:
    """连接池类，用于管理 RabbitMQ 连接"""

    def __init__(self, broker_address="127.0.0.1", broker_port=5672, pool_size=20):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.pool_size = pool_size
        self.connections = []
        self.lock = Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(self.pool_size):
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.broker_address, port=self.broker_port))
            self.connections.append(connection)

    @contextmanager
    def get_connection(self):
        """从连接池中获取一个连接"""
        with self.lock:
            if not self.connections:
                raise RuntimeError("Connection pool is empty.")
            connection = self.connections.pop(0)
        try:
            yield connection
        finally:
            with self.lock:
                self.connections.append(connection)


class CameraMQ:
    """
    每个 CameraMQ 实例都会创建自己的 RabbitMQ 连接。
    因为在 RabbitMQ 中，每个连接都是独立的，每个连接都有自己的通道。
    每个通道都有自己的队列和消费者。
    NOTE 可以使用连接池来管理连接。连接池可以重用连接，而不是为每个 Camera 实例创建新的连接。
    """

    def __init__(self, camera_id, connection_pool: ConnectionPool):
        self.camera_id = camera_id
        self.queue_name = f"camera_{camera_id}"
        self.connection_pool = connection_pool
        self.channel = None

    def _connect(self):
        """建立与 RabbitMQ 的连接并声明队列"""
        with self.connection_pool.get_connection() as connection:
            self.channel = connection.channel()
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            print(f"Camera {self.camera_id} connected and queue {self.queue_name} declared")

    def send_message(self, message):
        """向对应的队列发送消息"""
        if not self.channel or not self.channel.is_open:
            self._connect()
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2,  # 持久化消息
            )
        )
        print(f"Sent message to camera {self.camera_id}: {message}")

    @contextmanager
    def connect(self):
        """上下文管理器用于自动管理连接"""
        try:
            self._connect()
            yield self.channel
        finally:
            self.channel = None  # 释放通道，但不关闭连接（由连接池管理）

    def close_connection(self):
        """关闭 RabbitMQ 连接"""
        if self.channel and self.channel.is_open:
            self.channel.close()
            print(f"Camera {self.camera_id} channel closed")





class Consumer:
    def __init__(self, camera_mq: CameraMQ, max_retries=5, retry_delay=5):
        self.camera_mq = camera_mq
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @contextmanager
    def _managed_connection(self):
        """内部使用的上下文管理器，用于自动管理连接"""
        retries = 0
        while retries < self.max_retries or self.max_retries == -1:  # -1 表示无限重试
            try:
                with self.camera_mq.connect() as channel:
                    yield channel
                break  # 如果连接成功，退出循环
            except AMQPConnectionError as e:
                print(f"Connection error: {e}. Retrying in {self.retry_delay} seconds...")
                retries += 1
                time.sleep(self.retry_delay)
            except Exception as e:
                print(f"Unexpected error: {e}. Stopping consumer.")
                break
        else:
            print(f"Max retries reached. Unable to establish connection for camera {self.camera_mq.camera_id}.")
            raise ConnectionError("Unable to establish connection after multiple retries.")


    def consume_message(self, timeout=None):
        """消费一条消息并返回"""
        with self._managed_connection() as channel:
            method_frame, header_frame, body = channel.basic_get(queue=self.camera_mq.queue_name, auto_ack=True)
            if method_frame:
                return body.decode()  # 返回解码后的消息
            else:
                print(f"No message received from camera {self.camera_mq.camera_id}.")
                return None


# 示例用法
if __name__ == "__main__":
    # 创建连接池
    connection_pool = ConnectionPool(broker_address="127.0.0.1", broker_port=5672, pool_size=10)
    # 创建20个摄像头实例
    cameras = [CameraMQ(i, connection_pool) for i in range(1, 21)]
    # 为每个摄像头创建一个消费者实例
    consumers = [Consumer(camera, max_retries=-1, retry_delay=5) for camera in cameras]
    # 消费消息
    for i, consumer in enumerate(consumers):
        message = consumer.consume_message(timeout=2)  # 等待最多10秒
        if message:
            print(f"Received message from camera {i + 1}: {message}")
        else:
            print(f"No message received from camera {i + 1} within the timeout period.")