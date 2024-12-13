# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/13 下午9:08
@Author  : Kend
@FileName: test_use_mq.py
@Software: PyCharm
@modifier:
"""

# 假设您已经有一个 SQL 类和 MQ 类的实现
from collector import  PerformanceMonitor
from your_mq_module import MQ, SQL



if __name__ == "__main__":
    # 初始化 SQL 和 MQ 实例
    sql_instance = SQL()  # 请替换为您的 SQL 类实例化方式
    mq_instance = MQ()   # 请替换为您的 MQ 类实例化方式
    # 创建性能监控器实例，使用 RabbitMQ
    monitor = PerformanceMonitor(mq=mq_instance, sql=sql_instance, use_process_queue=False)
    # 启动监控
    monitor.start_monitoring()


    # ------------------使用进程队列------------------
    # 假设您已经有一个 SQL 类和 MQ 类的实现
    # from your_sql_module import SQL
    # from your_mq_module import ProcessQueue  # 假设您有一个 ProcessQueue 类
    if __name__ == "__main__":
        # 初始化 SQL 和 ProcessQueue 实例
        sql_instance = SQL()  # 请替换为您的 SQL 类实例化方式
        process_queue_instance = ProcessQueue()  # 请替换为您的 ProcessQueue 类实例化方式
        # 创建性能监控器实例，使用进程队列
        monitor = PerformanceMonitor(mq=process_queue_instance, sql=sql_instance, use_process_queue=True)
        # 启动监控
        monitor.start_monitoring()
