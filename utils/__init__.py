# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:36
@Author  : Kend
@FileName: __init__.py.py
@Software: PyCharm
@modifier:
"""
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)

from utils.logger import logger
from utils.timer import MyTimer as timer

__all__ = ['logger', 'timer']
