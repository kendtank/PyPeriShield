# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 下午8:42
@Author  : Kend
@FileName: track_manager.py
@Software: PyCharm
@modifier:
"""
import datetime
import cv2
import numpy as np
from core.trajectory.track import Track
from tools import logger


# 轨迹管理器-以摄像头为单位
class TracksManager:
    def __init__(self, max_missed_frames=30):
        self.tracks = {}  # 轨迹实例
        self.max_missed_frames = max_missed_frames
        self.lines = []  # 存储给定的线段列表

    def set_lines(self, lines):
        """设置要检测相交的线段"""
        self.lines = lines

    """处理每一帧"""
    def process_frame(self, onlines_id, onlines_tlwh):
        # 先处理id,以及把轨迹点更新了
        for i, id in enumerate(onlines_id):
            # 如果在现有的轨迹就增加轨迹点
            if id in self.tracks:
                self.tracks[id].update(onlines_tlwh[i])
            else:
                # 如果不在现有的轨迹就创建新的轨迹对象，并新增
                self.tracks[id] = Track(id, self.max_missed_frames)
                self.tracks[id].update(onlines_tlwh[i])

        # 遍历现在所有的id轨迹对象, 对于当前丢失的记录丢失次数
        for track in self.tracks.values():
            # 如果不在当前帧的id中，则未检测到，未检测帧数+1
            if track.id not in onlines_id:
                track.increment_missed_frames()


        # 继续遍历所有的id轨迹对象，如果超出了阈值，则删除这个id轨迹对象
        ids_to_remove = [id for id, track in self.tracks.items() if track.is_stale()]
        for id in ids_to_remove:
            del self.tracks[id]

        # 继续遍历所有的id轨迹对象
        for track in self.tracks.values():
            # 遍历每个id轨迹对象，检查轨迹中两两的点是否存在与给定的lines相交
            if track.check_intersection(self.lines):
                print(f"Track ID {track.id} intersected with lines: {track.get_intersected_lines()}")
                # print(track.intersected_lines)  # {((800, 643), (1082, 604))}
                # 如果轨迹中两两的点存在与给定的lines相交, 判断另外一条的周界线这个id有没有触碰
                lines_set = set(tuple(tuple(sublist) for sublist in line) for line in self.lines)
                print(lines_set, track.intersected_lines)
                # 碰线超过1次
                if len(track.intersected_lines) >= 1:
                    # print("两个集合相等有人翻越围墙", datetime.now(), len(lines_set))
                    return track.id

    def get_all_tracks(self):
        return list(self.tracks.values())



# 绘制轨迹
def draw_trajectories(frame, tracks, lines=None, max_length=30, track_id=None):
    """
    Args:
        lines: 判定线
        frame: 当前帧的图像
        tracks: get_all_tracks(), 也就是所有的轨迹id对象
        max_length: 轨迹显示的最大长度,
        track_id: 违规id
    Returns:
    """
    if track_id is not None:
        cv2.putText(frame, f"ALARM: ID-{track_id} Invasion!!!",
                    org=(5, 60), fontScale=2, color=(0,0,255), thickness=2, fontFace=2)


    if lines is not None:
        _color = (0, 255, 0)  # 绿色  bgr
        # 显示 判定线
        cv2.polylines(frame, np.array(lines, dtype=np.int32), isClosed=False, color=_color, thickness=3)
        # cv2.polylines(frame, lines, isClosed=False, color=_color, thickness=3)
    # 遍历所有轨迹对象
    for track in tracks:
        # if len(track.locations) > 0 and current_frame >= len(track.locations):
        if len(track.locations) > 0:
            if len(track.locations) > max_length:
                points = np.array(track.locations[-max_length:], dtype=np.int32)  # 只显示最新的三十帧
            else:
                points = np.array(track.locations, dtype=np.int32)
            color = (0, 255, 0)  # 默认颜色为绿色

            # 如果轨迹与线段相交，改变颜色
            # print("track.intersected_lines:::", track.intersected_lines)  # {((1045, 445), (1085, 1079))}
            if track.intersected_lines:  # 非空
                # TODO 相交, 记录违规时间， 推送三维报警
                logger.info(f"人员翻墙！！！！, 时间为：{datetime.datetime.now()}")
                color = (0, 0, 255)  # 轨迹红色表示相交
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)


            # 在相交点处添加标记  相交的线变红色
            for line in track.intersected_lines:
                cv2.line(frame, tuple(map(int, line[0])), tuple(map(int, line[1])), (0, 0, 255), 5)

    return frame