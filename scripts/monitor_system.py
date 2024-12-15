#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/14
@Time: 08:25
@Description: monitor_system - 文件描述
@Modify:
@Contact: tankang0722@gmail.com


说明：
SystemMonitor类
    构造函数 (__init__):
        接收 host、port 和 interval 参数，用于设置 Web 服务地址、端口和监控数据的刷新间隔。
    setup_routes 方法:
        定义了 /metrics 路由，返回当前系统监控资源的json数据。
        / 返回html页面可视化   # 后续会改，统一又nginx去代理
    monitor_system 方法:
        在独立线程中采集系统的 CPU、内存、硬盘以及 GPU 信息。
    run 方法:
        启动 Web 服务和监控线程。
    stop 方法:
    停止监控服务。
注意：
    封装的监控服务是一个完全独立的模块，不需要与其他服务共享 FastAPI 实例。
    如果需要与其他服务共享 FastAPI 实例，建议改为使用全局实例的方式，并通过配置或依赖注入管理。
"""


import threading
import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from jinja2 import Template
import uvicorn



class SystemMonitor(threading.Thread):
    def __init__(self, host="127.0.0.1", port=8000, interval=5):
        """
        一个可独立运行的监控服务类.
        :param host: web服务的主机地址.
        :param port: 主机端口.
        :param interval: 更新监控数据的时间间隔.
        """
        super().__init__()
        self.host = host
        self.port = port
        self.interval = interval
        self.running = False
        # 创建FastAPI实例 如果系统中提供了外部的api，建议使用依赖注入
        self.app = FastAPI()


        # 初始化监控数据
        self.cpu_usage = []
        self.memory_info = {}
        self.disk_info = {}
        self.gpu_info = []

        # get方法的 路由
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/metrics")
        async def get_metrics():
            return JSONResponse(content={
                "cpu_usage": psutil.cpu_percent(percpu=False),  # 综合占用率
                "memory_info": self.format_memory(self.memory_info),
                "disk_info": self.format_memory(self.disk_info),
                "gpu_info": self.gpu_info
            })

        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            html_template = Template(
                """<!DOCTYPE html>
                <html>
                <head>
                    <title>System Monitor</title>
                    <style>
                        body { font-family: Arial, sans-serif; background-color: #1e1e1e; color: #c5c6c7; }
                        h1, h2 { text-align: center; color: #66fcf1; }
                        table { width: 90%; margin: auto; border-collapse: collapse; margin-bottom: 30px; }
                        th, td { border: 1px solid #45a29e; padding: 8px; text-align: center; }
                        th { background-color: #0b0c10; color: #66fcf1; }
                        tr:nth-child(even) { background-color: #2e2f30; }
                        tr:hover { background-color: #45a29e; color: #0b0c10; }
                    </style>
                </head>
                <body>
                    <h1>System Monitoring Dashboard</h1>
                    <h2>CPU Usage</h2>
                    <table>
                        <tr>
                            <th>Core</th>
                            <th>Usage (%)</th>
                        </tr>
                        {% for core, usage in cpu_usage %}
                        <tr>
                            <td>Core {{ core }}</td>
                            <td>{{ usage }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                    <h2>Memory Info</h2>
                    <table>
                        <tr>
                            <th>Total</th>
                            <th>Used</th>
                            <th>Free</th>
                            <th>Percent</th>
                        </tr>
                        <tr>
                            <td>{{ memory_info.total }}</td>
                            <td>{{ memory_info.used }}</td>
                            <td>{{ memory_info.free }}</td>
                            <td>{{ memory_info.percent }}%</td>
                        </tr>
                    </table>
                    <h2>Disk Info</h2>
                    <table>
                        <tr>
                            <th>Total</th>
                            <th>Used</th>
                            <th>Free</th>
                            <th>Percent</th>
                        </tr>
                        <tr>
                            <td>{{ disk_info.total }}</td>
                            <td>{{ disk_info.used }}</td>
                            <td>{{ disk_info.free }}</td>
                            <td>{{ disk_info.percent }}%</td>
                        </tr>
                    </table>
                    <h2>GPU Info</h2>
                    <table>
                        <tr>
                            <th>Name</th>
                            <th>Total Memory</th>
                            <th>Used Memory</th>
                            <th>Free Memory</th>
                            <th>GPU Utilization</th>
                            <th>Memory Utilization</th>
                        </tr>
                        {% for gpu in gpu_info %}
                        <tr>
                            <td>{{ gpu.name }}</td>
                            <td>{{ gpu.total_memory }}</td>
                            <td>{{ gpu.used_memory }}</td>
                            <td>{{ gpu.free_memory }}</td>
                            <td>{{ gpu.gpu_utilization }}%</td>
                            <td>{{ gpu.memory_utilization }}%</td>
                        </tr>
                        {% endfor %}
                    </table>
                </body>
                </html>
                """
            )
            return html_template.render(
                cpu_usage=enumerate(self.cpu_usage),
                memory_info=self.format_memory(self.memory_info),
                disk_info=self.format_memory(self.disk_info),
                gpu_info=self.gpu_info

            )


    def monitor_system(self):
        """收集系统的使用数据."""
        while self.running:
            # 这里监控每个核心的使用率
            self.cpu_usage = psutil.cpu_percent(percpu=True)
            mem = psutil.virtual_memory()
            self.memory_info = {"total": mem.total, "used": mem.used, "free": mem.free, "percent": mem.percent}
            disk = psutil.disk_usage('/')
            self.disk_info = {"total": disk.total, "used": disk.used, "free": disk.free, "percent": disk.percent}
            # GPU info (if available), no gpu
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_info = []
                for i in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    self.gpu_info.append({
                        # "name": gpu_name.decode('utf-8'),
                        "name": gpu_name,
                        "total_memory": mem_info.total,
                        "used_memory": mem_info.used,
                        "free_memory": mem_info.free,
                        "gpu_utilization": util.gpu,
                        "memory_utilization": util.memory,
                    })

                pynvml.nvmlShutdown()

                # 默认有GPU
                try:
                    self.gpu_info = self.format_gpu_info(self.gpu_info)
                    # 验证有直接不管他，转换必定为正常的
                except:
                    # 没有不管直接显示[]
                    pass

            except ImportError:
                self.gpu_info = [{"name": "N/A", "total_memory": "N/A", "used_memory": "N/A", "free_memory": "N/A",
                                  "gpu_utilization": "N/A", "memory_utilization": "N/A"}]

            threading.Event().wait(self.interval)


    # 内存信息 GB
    def format_memory(self, info):
        """Format memory data in GB."""
        print('info:::', info)
        # return {k: f"{v / (1024 ** 3):.2f} GB" if isinstance(v, (int, float)) else v for k, v in info.items()}
        return {k: f"{v / (1024 ** 3):.2f} GB" if isinstance(v, int) else v for k, v in info.items()}

    # gpu MB
    def format_gpu_info(self, info):
        """Format GPU memory data."""
        for gpu in info:
            gpu['total_memory'] = f"{gpu['total_memory'] / (1024 ** 2):.2f} MB"
            gpu['used_memory'] = f"{gpu['used_memory'] / (1024 ** 2):.2f} MB"
            gpu['free_memory'] = f"{gpu['free_memory'] / (1024 ** 2):.2f} MB"
        return info
    def run(self):
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
        monitor_thread.daemon = True  # 设置为守护线程，主线程异常后 守护线程会自动终止。
        monitor_thread.start()


        uvicorn.run(self.app, host=self.host, port=self.port)

    def stop(self):
        """停止监控服务"""
        self.running = False


# Example
if __name__ == "__main__":
    monitor = SystemMonitor(host="127.0.0.1", port=8000, interval=5)
    monitor.start()

    try:
        monitor.join()
    except KeyboardInterrupt:
        monitor.stop()
        print("监控服务停止.")
