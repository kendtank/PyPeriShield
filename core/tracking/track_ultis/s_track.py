#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:54
@Description: s_track - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""

from core.tracking.track_ultis.base_track import BaseTrack, TrackState
from core.tracking.track_ultis.kalman_filter import KalmanFilter
import numpy as np


""" STrack对象 是一个用于目标跟踪的类，主要使用卡尔曼滤波器来估计目标的状态 """
class STrack(BaseTrack):
    # shared_kalman 是一个静态属性，表示所有 STrack 实例共享同一个卡尔曼滤波器实例。
    shared_kalman = KalmanFilter()  # 卡尔曼滤波器用于估计目标的状态，从而提高跟踪的准确性。

    def __init__(self, tlwh, score):
        """
        Args:
            tlwh: 目标的初始边界框，格式为 (top left x, top left y, width, height)。
            score 是目标的置信度分数。
        function: STrack 类主要用于管理目标跟踪中的轨迹信息，包括轨迹的初始化、预测、更新和重新激活等功能。
        """
        # 等待被激活
        self._tlwh = np.asarray(tlwh, dtype=np.float64)   # 存储初始的边界框信息。
        self.kalman_filter = None   # 用于存储卡尔曼滤波器实例
        self.mean, self.covariance = None, None    # 分别存储卡尔曼滤波器的状态均值和协方差矩阵，初始为 None
        self.is_activated = False   # 表示该轨迹是否已激活，初始为 False
        self.score = score  # 存储目标的置信度分数
        self.tracklet_len = 0   # 记录轨迹的长度，初始为 0

    # 预测下一个时间的目标
    def predict(self):
        mean_state = self.mean.copy()   # mean_state 当前状态的副本
        # 如果当前轨迹状态不是 Tracked，则将 mean_state 的第8个元素（通常表示速度）设为 0
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        # 使用卡尔曼滤波器的 predict 方法更新 self.mean 和 self.covariance
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    # 多轨迹预测方法
    @staticmethod
    def multi_predict(stracks):
        # 批量预测多个轨迹的状态
        if len(stracks) > 0:    # 包含多个 STrack 实例的列表
            # multi_mean 和 multi_covariance 分别存储所有轨迹的状态均值和协方差矩阵
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            # 如果某个轨迹的状态不是 Tracked，则将 multi_mean 的第8个元素设为 0
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            # 使用共享的卡尔曼滤波器的 multi_predict 方法批量预测所有轨迹的状态
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            # 更新每个轨迹的 mean 和 covariance
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    # 用来激活一个新的轨迹
    def activate(self, kalman_filter, frame_id):
        """ 开始一个新的轨迹 """
        self.kalman_filter = kalman_filter # 卡尔曼滤波器实例
        # 获取一个新的轨迹ID
        self.track_id = self.next_id()
        # 卡尔曼滤波器的 initiate 方法初始化轨迹的状态均值和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        # 重置轨迹长度 tracklet_len 为 0。
        self.tracklet_len = 0
        # 设置轨迹状态为 Tracked。
        self.state = TrackState.Tracked
        # 如果当前帧编号为 1，则设置 is_activated 为 True。
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        # # 设置当前帧编号 frame_id 和轨迹起始帧编号 start_frame。
        self.frame_id = frame_id
        self.start_frame = frame_id

    """ 重新激活一个已经存在的轨迹 """
    def re_activate(self, new_track, frame_id, new_id=False):
        """
        Args:
            new_track:  new_track 是一个新的 STrack 实例，包含更新后的信息
            frame_id: 当前帧的编号。
            new_id: new_id 是一个布尔值，表示是否为轨迹分配一个新的ID。
        Returns:
        """
        # 使用卡尔曼滤波器的 update 方法更新轨迹的状态均值和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))
        # 重置轨迹长度 tracklet_len 为 0
        self.tracklet_len = 0
        # 设置轨迹状态为 Tracked。
        self.state = TrackState.Tracked
        self.is_activated = True
        # 设置当前帧编号 frame_id。
        self.frame_id = frame_id
        # # 如果 new_id 为 True，则使用 next_id 方法获取一个新的轨迹ID。
        if new_id:
            self.track_id = self.next_id()
        # # 更新轨迹的置信度分数 score。
        self.score = new_track.score

    # 更新一个匹配的轨迹
    def update(self, new_track, frame_id):
        """
        Args:
            new_track 是一个新的 STrack 实例，包含更新后的信息。
            frame_id: int 当前帧编号
        :return:
        """
        # 设置当前帧编号 frame_id
        self.frame_id = frame_id
        # 增加轨迹长度 tracklet_len
        self.tracklet_len += 1
        # 获取新的边界框信息 new_tlwh
        new_tlwh = new_track.tlwh
        # 使用卡尔曼滤波器的 update 方法更新轨迹的状态均值和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        # 设置轨迹状态为 Tracked
        self.state = TrackState.Tracked
        # 设置 is_activated 为 True
        self.is_activated = True
        # 更新轨迹的置信度分数 score
        self.score = new_track.score


    @property
    # 获取当前轨迹的边界框信息  # @property装饰器 -> 方法属性
    def tlwh(self):
        # 如果 self.mean 为 None，则返回初始的边界框信息 _tlwh。
        if self.mean is None:
            return self._tlwh.copy()
        # 否则，从卡尔曼滤波器的状态均值中提取边界框信息。
        ret = self.mean[:4].copy()
        # 将状态均值中的 (center_x, center_y, aspect_ratio, height) 转换为 (top_left_x, top_left_y, width, height) 格式。
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        # 返回转换后的边界框信息。
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)