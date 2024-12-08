###  Language Selection
[View Chinese Version](../README.md)
---

### Project Name
PyPeriShield: "PeriShield" combines "Perimeter" and "Shield" 
### Project Overview
**** Smart Intrusion Detection System
- topic: Distributed Cluster Inference, 3D Visualization, Trajectory Tracking, Intrusion Alerts, and Face Recognition
- Introduction: This project develops an advanced smart park security solution that integrates multiple technologies. It leverages distributed cluster inference for efficient parallel processing, handling large-scale data and complex scenarios. A 3D visualization platform provides real-time monitoring with intuitive views of personnel and vehicle activities. The system tracks trajectories in real-time and records violation videos for post-event review. Virtual fences trigger intrusion alerts for unauthorized entries, ensuring timely responses. Face recognition identifies registered individuals and alerts on unregistered ones. Suitable for smart parks, corporate campuses, and public spaces, this system enhances security management efficiency and safety.

Project Goals: 
Event accuracy rate higher than 95%
Less than 0.5 false alarms per hour
Event detection within 1 minute
Event traceability for up to 6 months

### Project Structure

#### 系统架构设计
1. 客户端入口
- 三维智慧园区：控制摄像头算法开关和参数配置，并展示和交互报警信息。
2. 海葵云分布式平台
- 负载均衡器：分发不同摄像头的任务到不同的算力集群节点。
3. 采集层
- RTSP/SDK服务器：通过大华或海康的SDK获取视频流，或者通过RTSP协议接收视频流。
- 消息队列：用于解耦视频流获取与后续处理，保证高并发下的稳定性。
4. 处理层
- GPU集群：
- 目标检测与跟踪服务：每台主机运行目标检测和跟踪算法。
- 轨迹管理与报警判断：实时管理目标轨迹，并根据预定义区域和判定线判断是否存在翻围墙行为。
5. 存储层
- 分布式文件系统/对象存储：保存生成的可视化视频片段。
- 关系型数据库：存储报警信息及相关元数据。
- 内存数据库（Redis）：快速存储和查询目标轨迹数据。
6. 交互层
- 消息队列：用于从报警判断模块向回放获取与推送服务发送消息。
- 回放获取与推送服务：从海康服务器下载视频片段并推送到三维智慧园区。
7. 监控与日志
- 监控系统：Prometheus + Grafana 实时监控系统性能。
- 日志管理系统：集中化日志管理。

#### 模块说明
##### `config/`
- **`settings.py`**: 系统配置文件，包含全局配置项。
- **`secrets.py`**: 敏感信息管理，如API密钥。仅在开发环境中使用，生产环境应通过环境变量或安全服务获取。

##### `core/`
- **`detection/`**: 目标检测相关逻辑。
- **`tracking/`**: 目标跟踪相关逻辑。
- **`trajectory/`**: 轨迹管理相关逻辑。
- **`alarm/`**: 报警相关逻辑。
- **`playback/`**: 回放视频片段获取和推送。

#### `data/`
- **`database/`**: 数据库操作。
- **`redis/`**: Redis缓存管理。
- **`storage/`**: 文件存储操作。

#### `services/`
- **`video_stream/`**: 视频流处理。
- **`message_queue/`**: 消息队列。
- **`api_gateway/`**: API。

#### `utils/`
- **`timer`**: 提供一些时间工具函数。
- **`logger/`**: 日志管理。
- **`metrics/`**: 性能监控。
- **`visualization/`**: 可视化逻辑。

#### `tests/`
- 单元测试和集成测试。

#### `scripts/`

#### `requirements.txt`
- 列出项目的依赖项。

#### `main.py`
- 项目的入口文件，负责启动各个组件和服务。

---

### 使用示例

在`main.py`中，您可以按照以下方式启动项目：

```python
from services.api_gateway.router import ApiGatewayRouter
from services.message_queue.consumer import MessageQueueConsumer
from services.video_stream.stream_handler import VideoStreamHandler
from config.settings import load_config

def main():
    config = load_config()

    # 初始化并启动API网关
    api_gateway = ApiGatewayRouter(config)
    api_gateway.start()

    # 初始化并启动消息队列消费者
    message_queue_consumer = MessageQueueConsumer(config)
    message_queue_consumer.start_consuming()

    # 初始化并启动视频流处理器
    video_stream_handler = VideoStreamHandler(config)
    video_stream_handler.start_processing()

    try:
        # 主循环，保持程序运行
        while True:
            pass
    except KeyboardInterrupt:
        # 清理资源
        api_gateway.stop()
        message_queue_consumer.stop_consuming()
        video_stream_handler.stop_processing()

if __name__ == "__main__":
    main()