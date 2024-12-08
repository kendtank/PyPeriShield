#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/7
@Time: 09:09
@Description: log_factory - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""


from .log_handlers import setup_logger

"""
统一使用同一个 logger 实例：通过单例模式确保在整个项目中使用同一个 logger 实例。
多线程和多进程支持：使用 enqueue=True 确保日志记录在多线程和多进程环境中安全可靠。
集中管理配置：所有日志相关的配置（格式化器、处理器等）都在一个地方进行，便于管理和维护。
"""


class LoggerFactory:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerFactory, cls).__new__(cls)
            cls._instance.logger = setup_logger()
        return cls._instance

    def get_logger(self):
        return self._instance.logger


# 获取日志记录器实例
logger_factory = LoggerFactory()
logger = logger_factory.get_logger()