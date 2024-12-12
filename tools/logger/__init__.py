#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/7
@Time: 09:09
@Description: __init__.py - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""

# 初始化日志记录器，并提供一个全局的 logger 实例。
from .log_factory import logger

__all__ = ['logger']
