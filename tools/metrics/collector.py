# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/12 下午9:08
@Author  : Kend
@FileName: collector.py
@Software: PyCharm
@modifier:
"""


import psutil
import pynvml
import platform
import os
import json
from datetime import datetime, timedelta
import threading
import schedule
import time


class PerformanceMonitor:
    def __init__(self, mq, sql, use_process_queue=False, write_times=None, interval=5 * 60):
        """
        初始化性能监控器

        :param mq: 消息队列实例 (RabbitMQ 或 ProcessQueue)
        :param sql: SQL 数据库实例
        :param use_process_queue: 是否使用进程队列，默认为 False (使用 RabbitMQ)
        :param write_times: 写入数据库的时间点列表，默认为每天的 8:00, 12:00, 16:00, 20:00, 00:00, 04:00
        :param interval: 监控间隔时间（秒），默认为5分钟
        """
        self.mq = mq
        self.sql = sql
        if write_times is None:
            write_times = [
                datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),
                datetime.now().replace(hour=12, minute=0, second=0, microsecond=0),
                datetime.now().replace(hour=16, minute=0, second=0, microsecond=0),
                datetime.now().replace(hour=20, minute=0, second=0, microsecond=0),
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                datetime.now().replace(hour=4, minute=0, second=0, microsecond=0)
            ]
        self.write_times = write_times
        self.interval = interval
        self.hostname = platform.node()
        self.pid = os.getpid()

        # 设置定时任务
        for wt in self.write_times:
            schedule.every().day.at(wt.strftime('%H:%M')).do(self.monitor_and_send)

        # 启动调度器线程
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()

    def get_cpu_memory_usage(self):
        """获取CPU和内存使用情况"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        return {
            'cpu_usage': cpu_usage,
            'memory_used': memory_info.used / (1024 ** 3),
            'memory_available': memory_info.available / (1024 ** 3)
        }

    def get_disk_usage(self):
        """获取磁盘使用情况"""
        disk_usage = psutil.disk_usage('/')
        return {
            'disk_used': disk_usage.used / (1024 ** 3),
            'disk_free': disk_usage.free / (1024 ** 3)
        }

    def get_gpu_usage(self):
        """获取GPU使用情况"""
        try:
            pynvml.nvmlInit()
            deviceCount = pynvml.nvmlDeviceGetCount()
            gpu_usages = []
            for i in range(deviceCount):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usages.append({
                    'gpu_index': i,
                    'gpu_used_memory': info.used / (1024 ** 2),
                    'gpu_total_memory': info.total / (1024 ** 2),
                    'gpu_utilization': util.gpu,
                    'gpu_memory_utilization': util.memory
                })
            pynvml.nvmlShutdown()
            return gpu_usages
        except pynvml.NVMLError:
            return []

    def should_write_to_db(self):
        """检查当前时间是否在写入时间点附近"""
        now = datetime.now()
        for wt in self.write_times:
            if now - timedelta(minutes=5) <= wt.replace(day=now.day) <= now + timedelta(minutes=5):
                return True
        return False

    def monitor_and_send(self):
        """收集性能数据并发送到消息队列"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'hostname': self.hostname,
            'pid': self.pid,
            'cpu_memory': self.get_cpu_memory_usage(),
            'disk': self.get_disk_usage(),
            'gpu': self.get_gpu_usage()
        }
        message = json.dumps(data)
        if self.should_write_to_db():
            self.mq.send_message(message)

    def run_scheduler(self):
        """运行调度器"""
        while True:
            schedule.run_pending()
            time.sleep(1)

    def start_monitoring(self):
        """启动监控"""
        print("Performance monitoring started.")
        try:
            while True:
                time.sleep(self.interval)
                self.monitor_and_send()
        except KeyboardInterrupt:
            print("Performance monitoring stopped.")

