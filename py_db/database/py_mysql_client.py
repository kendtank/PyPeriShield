# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:46
@Author  : Kend
@FileName: py_mysql_client.py
@Software: PyCharm
@modifier:
"""

import pymysql
import aiomysql
import asyncio
import threading
from contextlib import asynccontextmanager
from functools import wraps
from tools.logger import logger


"""
代码说明:
    MySQLClient 类：
        该类封装了 PyMySQL 和 aiomysql 的功能，提供了同步和异步两种接口。
        __init__ 方法初始化数据库连接参数，并设置默认的最大连接池大小为 10。
        _create_pool 和 _create_sync_connection 方法分别用于创建异步连接池和同步连接。
        get_async_connection 和 get_sync_connection 方法分别用于获取异步和同步连接，并确保每个线程有自己的连接。
        execute_query_async 和 execute_query_sync 方法用于执行查询操作。
        execute_write_async 和 execute_write_sync 方法用于执行插入、更新或删除操作。
        _reconnect 和 _reconnect_sync 方法实现了重连接机制，确保在连接丢失时能够自动恢复。
        close 方法用于关闭所有连接，包括异步连接池和同步连接。
    装饰器：
        @async_context_manager 和 @sync_context_manager 分别用于装饰异步和同步方法，确保在发生数据库错误时能够自动重连。
    多线程支持：
        使用 threading.local() 来为每个线程维护独立的同步连接，确保线程安全。
    异步支持：
        使用 aiomysql 实现异步操作，结合 asyncio 提供异步查询和写入功能。
    日志记录：
        使用 封装的多线程 logger 模块记录连接创建、重连等关键操作的日志，方便调试和监控。
使用示例:
    在多进程环境中，每个进程可以创建自己的 MySQLClient 实例，并根据需要选择同步或异步接口。
    你可以通过 execute_query_sync 和 execute_write_sync 方法进行同步操作，或者通过 execute_query_async 和 execute_write_async 方法进行异步操作。
    重连接机制会在连接丢失时自动尝试重新连接，确保应用的高可用性。
    见代码底部
注意事项:
    NOTE:
    连接池大小：根据你的应用负载和数据库服务器的配置，合理调整 maxsize 参数，以确保最佳性能。由于我的算法检测平台与sql库的交互，所以默认10
    在多线程环境中，虽然你可以只实例化一次 MySQLClient 类，但为了确保线程安全和性能优化，每个线程应该有自己的数据库连接。这是因为数据库连接本身不是线程安全的，多个线程共享同一个连接可能会导致竞态条件或其他并发问题。

事务管理：
    如果你需要更复杂的事务管理（如多条语句的原子性），可以在 execute_write_sync 和 execute_write_async 方法中加入事务控制。
错误处理：
    在实际应用中，建议进一步完善错误处理逻辑，特别是在网络不稳定或数据库服务器出现故障的情况下。


线程安全的实现方式
线程本地存储（Thread-Local Storage）：
在 MySQLClient 类中，我们使用了 threading.local() 来为每个线程维护独立的同步连接。这样，每个线程都有自己独立的数据库连接，避免了线程间的竞争。
异步连接池：
对于异步操作，我们使用了 aiomysql 的连接池。连接池允许多个协程共享一组连接，而不会阻塞其他协程。连接池会自动管理连接的创建、释放和重用，确保高并发场景下的性能。
实例化方式

"""




"""
在多线程编程中，如果多个线程需要共享某些数据，但又希望这些数据对于每个线程来说是独立的，那么就可以使用线程局部数据。
这有助于避免线程安全问题， 每个线程操作的是自己的数据副本，而不是共享的数据。

作用：
    当创建一个 threading.local() 实例时，你实际上创建了一个可以在线程之间共享的对象，但是每个线程对这个对象的属性访问都是独立的。
    也就是说，每个线程都可以设置和获取该对象的属性，而不会影响到其他线程看到的值。
    threading.local() 提供了一种简单且有效的方式来管理线程之间的独立状态，同时避免了复杂的锁机制或同步问题。
"""

# 线程本地存储，用于存储每个线程的数据库连接
thread_local = threading.local()




# 在每个线程中维护独立的数据库连接。
class MySQLClient:
    def __init__(self, host, port, user, password, db, charset='utf8mb4', maxsize=10):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.charset = charset
        self.maxsize = maxsize
        self.pool = None  # 异步连接池
        self.sync_conn = None  # 同步连接（线程本地）


    # 异步上下文管理器装饰器
    def async_context_manager(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not self.pool:
                await self._create_pool()
            try:
                async with func(self, *args, **kwargs) as result:
                    yield result
            except pymysql.MySQLError as e:
                logger.error(f"Database error: {e}")
                await self._reconnect()
                raise
        return wrapper


    # 创建异步连接池
    async def _create_pool(self):
        if not self.pool:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db,
                charset=self.charset,
                maxsize=self.maxsize,
                loop=asyncio.get_event_loop()
            )
            logger.info("Created async connection pool.")


    # 获取异步连接
    @async_context_manager
    async def get_async_connection(self):
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            self.pool.release(conn)


    # 执行异步查询
    async def execute_query_async(self, query, params=None):
        async with self.get_async_connection() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, params)
                result = await cursor.fetchall()
                return result


    # 执行异步插入/更新/删除
    async def execute_write_async(self, query, params=None):
        async with self.get_async_connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                await conn.commit()
                return cursor.rowcount


    # 同步上下文管理器装饰器
    def sync_context_manager(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(thread_local, 'conn') or thread_local.conn is None:
                self._create_sync_connection()
            try:
                with func(self, *args, **kwargs) as result:
                    yield result
            except pymysql.MySQLError as e:
                logger.error(f"Database error: {e}")
                self._reconnect_sync()
                raise
        return wrapper


    # 创建同步连接
    def _create_sync_connection(self):
        thread_local.conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db=self.db,
            charset=self.charset,
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info("Created sync connection for thread %s.", threading.current_thread().name)


    # 获取同步连接
    @sync_context_manager
    def get_sync_connection(self):
        yield thread_local.conn


    # 执行同步查询
    def execute_query_sync(self, query, params=None):
        with self.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchall()
                return result


    # 执行同步插入/更新/删除
    def execute_write_sync(self, query, params=None):
        with self.get_sync_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount


    # 重连机制（异步）
    async def _reconnect(self):
        logger.info("Attempting to reconnect asynchronously...")
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
        await self._create_pool()
        logger.info("Reconnected asynchronously.")


    # 重连机制（同步）
    def _reconnect_sync(self):
        logger.info("Attempting to reconnect synchronously...")
        if hasattr(thread_local, 'conn') and thread_local.conn:
            thread_local.conn.close()
            thread_local.conn = None
        self._create_sync_connection()
        logger.info("Reconnected synchronously.")


    # 关闭所有连接
    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("Closed async connection pool.")
        if hasattr(thread_local, 'conn') and thread_local.conn:
            thread_local.conn.close()
            logger.info("Closed sync connection for thread %s.", threading.current_thread().name)




# 示例用法
async def example_usage():
    client = MySQLClient(host='localhost', port=3306, user='root', password='password', db='test_db')

    # 同步查询
    result = client.execute_query_sync("SELECT * FROM users LIMIT 1")
    print("Sync Query Result:", result)

    # 异步查询
    async_result = await client.execute_query_async("SELECT * FROM users LIMIT 1")
    print("Async Query Result:", async_result)

    # 关闭连接
    await client.close()



if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())


# ——————————————————————————注意——————————————————————————

"""
1. 单个实例化 + 线程本地连接
你可以只实例化一次 MySQLClient 类，并通过 threading.local() 为每个线程分配独立的连接。
这种方式下，你不需要为每个线程单独实例化 MySQLClient，但每个线程会有自己的连接。
"""

# 单个实例化
client = MySQLClient(host='localhost', port=3306, user='root', password='password', db='test_db')

# 多线程环境下使用
def thread_function():
    # 每个线程都会有自己的同步连接
    result = client.execute_query_sync("SELECT * FROM users LIMIT 1")
    print(f"Thread {threading.current_thread().name} Result: {result}")

# 创建多个线程
threads = []
for i in range(5):
    t = threading.Thread(target=thread_function)
    threads.append(t)
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()


"""
2. 每个线程独立实例化如果你希望每个线程都有自己的 MySQLClient 实例，你可以在每个线程中单独实例化 MySQLClient。
这种方式可以进一步隔离线程之间的资源，但可能会增加内存开销。
"""

def thread_function():
    # 每个线程独立实例化 MySQLClient
    client = MySQLClient(host='localhost', port=3306, user='root', password='password', db='test_db')
    result = client.execute_query_sync("SELECT * FROM users LIMIT 1")
    print(f"Thread {threading.current_thread().name} Result: {result}")
    asyncio.run(client.close())  # 关闭连接

# 创建多个线程
threads = []
for i in range(5):
    t = threading.Thread(target=thread_function)
    threads.append(t)
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()


"""
我在这里推荐使用第一种方式，即单个实例化 + 线程本地连接。这种方式既保持了代码的简洁性，又确保了线程安全和性能优化。具体来说：

单个实例化：你只需要在应用启动时创建一个 MySQLClient 实例，减少了对象创建的开销。
线程本地连接：通过 threading.local()，每个线程都有自己的数据库连接，避免了线程间的竞争和潜在的并发问题。
连接池：对于异步操作，aiomysql 的连接池可以有效地管理多个协程的连接，确保高并发场景下的性能。


为什么不能共享同一个连接？之前项目都是用的单例模式+锁。这种方式第一次尝试。
共享同一个数据库连接会导致以下问题：
竞态条件：多个线程同时访问同一个连接可能会导致数据不一致或损坏。
事务冲突：如果多个线程在同一连接上执行不同的事务，可能会导致事务隔离级别问题，甚至引发死锁。
性能瓶颈：即使没有明显的错误，共享连接也会成为性能瓶颈，因为每次操作都需要等待前一个操作完成。
总结
单个实例化 + 线程本地连接 是最推荐的方式，它既保证了线程安全，又避免了不必要的资源浪费。
异步连接池 可以有效管理多个协程的连接，适合高并发场景。
每个线程独立实例化 虽然也可以工作，但通常没有必要，除非你有特殊的需求或非常复杂的场景。(大多数情况把一个分布式容器只需要实例化一次连接)
"""