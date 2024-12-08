#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/7
@Time: 09:09
@Description: formatters - log_formatter
@Modify:
@Contact: tankang0722@gmail.com
"""


def get_log_format():
    return "{time:YYYY-MM-DD HH:mm:ss} | {level} | {process.name} | {thread.name} | {module}:{function}:{line} - {message}"

def get_level_color(level):
    colors = {
        "DEBUG": "\033[34m",    # 蓝色
        "INFO": "\033[32m",     # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",    # 红色
        "CRITICAL": "\033[95m"  # 洋红色
    }
    return colors.get(level, "\033[0m")  # 默认白色

def get_reset_color():
    return "\033[0m"