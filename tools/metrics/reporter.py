import threading
import psutil
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from jinja2 import Template
import uvicorn

class SystemMonitor(threading.Thread):
    def __init__(self, host="127.0.0.1", port=8000, interval=5):
        """
        System monitoring service with a web interface.

        :param host: Host for the web server.
        :param port: Port for the web server.
        :param interval: Interval for updating system stats in seconds.
        """
        super().__init__()
        self.host = host
        self.port = port
        self.interval = interval
        self.running = False
        self.app = FastAPI()

        # Initialize metrics
        self.cpu_usage = []
        self.memory_info = {}
        self.disk_info = {}
        self.gpu_info = []

        # Setup routes
        self.setup_routes()

    def setup_routes(self):
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
                memory_info=self.memory_info,
                disk_info=self.disk_info,
                gpu_info=self.gpu_info
            )

    def monitor_system(self):
        """Collect system metrics."""
        while self.running:
            # CPU usage
            self.cpu_usage = psutil.cpu_percent(percpu=True)

            # Memory info
            mem = psutil.virtual_memory()
            self.memory_info = {
                "total": mem.total,
                "used": mem.used,
                "free": mem.free,
                "percent": mem.percent,
            }

            # Disk info
            disk = psutil.disk_usage('/')
            self.disk_info = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }

            # GPU info (if available)
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
                        "name": gpu_name.decode('utf-8'),
                        "total_memory": mem_info.total,
                        "used_memory": mem_info.used,
                        "free_memory": mem_info.free,
                        "gpu_utilization": util.gpu,
                        "memory_utilization": util.memory,
                    })

                pynvml.nvmlShutdown()
            except ImportError:
                self.gpu_info = [{"name": "N/A", "total_memory": "N/A", "used_memory": "N/A", "free_memory": "N/A", "gpu_utilization": "N/A", "memory_utilization": "N/A"}]

            # Wait before collecting the next metrics
            threading.Event().wait(self.interval)

    def run(self):
        """Start the monitoring service."""
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_system)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Start the FastAPI server
        uvicorn.run(self.app, host=self.host, port=self.port)

    def stop(self):
        """Stop the monitoring service."""
        self.running = False

# Example usage
if __name__ == "__main__":
    monitor = SystemMonitor(host="127.0.0.1", port=8000, interval=5)
    monitor.start()

    try:
        monitor.join()
    except KeyboardInterrupt:
        monitor.stop()
        print("Monitoring service stopped.")
