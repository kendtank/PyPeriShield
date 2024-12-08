#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/7
@Time: 09:09
@Description: log_handler - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# print(project_root)  # D:\kend\WorkProject\PyPeriShield
os.chdir(project_root)
import os
from loguru import logger
from utils.logger.log_formatters import get_log_format, get_level_color, get_reset_color
from datetime import datetime


def setup_logger():
    log_format = get_log_format()

    # 添加控制台处理器
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=lambda record: f"{get_level_color(record['level'].name)}{log_format}{get_reset_color()}",
        level="DEBUG",
        enqueue=True  # 启用队列，确保多线程和多进程安全
    )

    log_path = "log"
    # 确保日志目录存在
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 动态生成日志文件路径，包含日期信息
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(log_path, current_date, "app_{time}.log")
    # 确保日期文件夹存在
    date_folder = os.path.join(log_path, current_date)
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)


    # 添加文件处理器，每天旋转一次
    logger.add(
        sink=log_file_path,
        format=log_format,
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        enqueue=True  # 启用队列，确保多线程和多进程安全
    )

    return logger


