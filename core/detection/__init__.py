# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:39
@Author  : Kend
@FileName: __init__.py.py
@Software: PyCharm
@modifier:
"""

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)


from core.detection.detector import Detector


__all__ = ['Detector']