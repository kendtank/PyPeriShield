# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:42
@Author  : Kend
@FileName: track.py
@Software: PyCharm
@modifier:
"""
import time
from shapely.geometry import LineString

# 轨迹存放类-按照id存放轨迹，一个id对应一个轨迹，对应一个实例
class Track:
    def __init__(self, id, max_missed_frames=30):
        self.id = id
        self.locations = []  # 记录每次更新的位置（中心点坐标）
        self.last_seen = time.time()  # 上次看到该目标的时间
        self.max_missed_frames = max_missed_frames  # 允许的最大未检测帧数与追终器保持一致
        self.missed_frames = 0  # 当前连续未检测帧数
        self.intersected_lines = set()  # 记录相交的线段
        # self.intersected_lines_mark = set() #


    def update(self, tlwh):
        current_time = time.time()
        center_x = tlwh[0] + tlwh[2] / 2
        center_y = tlwh[1] + tlwh[3] / 2
        self.locations.append((center_x, center_y))
        self.last_seen = current_time
        self.missed_frames = 0  # 重置未检测帧数计数器

    """丢失的帧自增1"""
    def increment_missed_frames(self):
        self.missed_frames += 1

    """检查轨迹是否过期"""
    def is_stale(self):
        return self.missed_frames > self.max_missed_frames

    """检查多线段轨迹是否与给定的线段相交"""
    def check_intersection(self, lines):
        """检查轨迹的所有点连接成的多段线是否与给定的线段相交"""
        if len(self.locations) < 2:
            return False

        # 将所有轨迹点连接成一条多段线
        polyline = LineString(self.locations)

        for line in lines:
            # 创建给定的线段
            segment = LineString([line[0], line[1]])

            # 检查多段线是否与当前线段相交
            if polyline.intersects(segment):
                # 把相交的周界线记录
                self.intersected_lines.add(tuple(tuple(sublist) for sublist in line))
                # self.intersected_lines.add(line)
                return True

        return False

    def get_intersected_lines(self):
        """返回与该轨迹相交的所有线段"""
        return list(self.intersected_lines)