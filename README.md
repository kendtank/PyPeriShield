## 项目名称
PyPeriShield “Py”代表Python编程语言，“PeriShield”结合了“Perimeter”（周界）和“Shield”（盾牌），强调系统提供了全面的周界防护。 
利元亨周界围墙入侵检测系统
项目简介：基于深度学习算法和实时视频流处理的智能安防系统，用于监测和检测利元亨公司产业园围墙周界区域是否存在翻越围墙行为。
项目目标：事件准确率高于95%，每小时误报率低于0.5次，事件及时性小于1分钟，事件可追溯6个月。
## 项目结构
### 架构设计
1. 客户端入口
- 三维智慧园区：控制摄像头算法开关，并展示和交互报警信息。
2. 海葵云分布式平台
- 负载均衡器：分发不同摄像头的任务到不同的算力集群节点。
3. 采集层
- RTSP/SDK服务器：通过大华或海康的SDK获取视频流，或者通过RTSP协议接收视频流。
- 消息队列：用于解耦视频流获取与后续处理，保证高并发下的稳定性。
4. 处理层
- GPU集群：
- 目标检测与跟踪服务：每台主机运行YOLOv8和ByteTrack进行目标检测和跟踪。
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
- 日志管理系统：ELK Stack 集中化日志管理。

### 模块说明
#### `config/`
- **`__init__.py`**: 初始化文件。
- **`settings.py`**: 系统配置文件，包含全局配置项。
- **`secrets.py`**: 敏感信息管理，如API密钥。仅在开发环境中使用，生产环境应通过环境变量或安全服务获取。

#### `core/`
- **`__init__.py`**: 初始化文件。
- **`detection/`**: 目标检测相关逻辑。
  - **`detector.py`**: 实现目标检测的具体逻辑。
  - **`model_loader.py`**: 负责加载和管理深度学习模型。
  - **`utils.py`**: 提供图像预处理等辅助工具。
- **`tracking/`**: 目标跟踪相关逻辑。
  - **`tracker.py`**: 实现目标跟踪的具体逻辑。
  - **`utils.py`**: 提供轨迹平滑等辅助工具。
- **`trajectory/`**: 轨迹管理相关逻辑。
  - **`manager.py`**: 负责轨迹的管理和存储。
  - **`utils.py`**: 提供轨迹存储格式等辅助工具。
- **`alarm/`**: 报警相关逻辑。
  - **`evaluator.py`**: 实现报警评估的具体逻辑。
  - **`notifier.py`**: 实现报警通知的具体逻辑。
  - **`utils.py`**: 提供报警条件定义等辅助工具。
- **`playback/`**: 回放视频片段获取和推送。
  - **`fetcher.py`**: 实现回放视频片段获取的具体逻辑。
  - **`pusher.py`**: 实现视频片段推送的具体逻辑。
  - **`utils.py`**: 提供视频编码/解码等辅助工具。

#### `data/`
- **`__init__.py`**: 初始化文件。
- **`database/`**: 数据库操作。
  - **`models.py`**: 定义ORM模型。
  - **`migrations/`**: 数据库迁移脚本。
  - **`queries.py`**: 提供数据查询逻辑。
- **`redis/`**: Redis缓存管理。
  - **`client.py`**: Redis客户端。
  - **`cache.py`**: 提供缓存管理逻辑。
- **`storage/`**: 文件存储操作。
  - **`s3.py`**: S3对象存储操作。
  - **`local.py`**: 本地文件系统操作。
  - **`utils.py`**: 提供存储辅助工具。

#### `services/`
- **`__init__.py`**: 初始化文件。
- **`video_stream/`**: 视频流处理。
  - **`stream_handler.py`**: 处理视频流的逻辑。
  - **`rtsp_client.py`**: RTSP协议客户端。
  - **`sdk_client.py`**: 大华/海康SDK客户端。
- **`message_queue/`**: 消息队列。
  - **`producer.py`**: 生产者逻辑。
  - **`consumer.py`**: 消费者逻辑。
  - **`utils.py`**: 提供队列辅助工具。
- **`api_gateway/`**: API网关。
  - **`router.py`**: 请求路由逻辑。
  - **`middleware.py`**: 中间件逻辑，如身份验证。
  - **`utils.py`**: 提供API网关辅助工具。

#### `utils/`
- **`__init__.py`**: 初始化文件。
- **`helpers.py`**: 提供各种辅助函数。
- **`logger/`**: 日志管理。
  - **`factory.py`**: 日志记录器工厂。
  - **`handlers.py`**: 日志处理器。
  - **`formatters.py`**: 日志格式化器。
- **`metrics/`**: 性能监控。
  - **`collector.py`**: 指标收集。
  - **`reporter.py`**: 指标报告。
  - **`utils.py`**: 提供监控辅助工具。

#### `tests/`
- **`__init__.py`**: 初始化文件。
- **`conftest.py`**: 测试配置。
- **`fixtures/`**: 测试夹具。
- **`integration/`**: 集成测试。
  - **`test_services.py`**: 服务集成测试。
- **`unit/`**: 单元测试。
  - **`test_core.py`**: 核心模块单元测试。
  - **`test_data.py`**: 数据层单元测试。
  - **`test_services.py`**: 服务层单元测试。

#### `scripts/`
- **`__init__.py`**: 初始化文件。
- **`setup_db.py`**: 数据库初始化脚本。
- **`manage.py`**: 项目管理脚本。

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