#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:54
@Description: byte_tracker - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""

import numpy as np
from core.tracking.track_ultis import matching
from core.tracking.track_ultis.s_track import STrack
from core.tracking.track_ultis.base_track import TrackState
from core.tracking.track_ultis.kalman_filter import KalmanFilter


""" ByteTrack 算法"""
class BYTETracker(object):

    """用于多目标跟踪任务，基于检测框的跟踪"""
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]  # 存储当前帧中活跃的跟踪对象。
        self.lost_stracks = []  # type: list[STrack]    # 存储当前帧中丢失的跟踪对象。
        self.removed_stracks = []  # type: list[STrack]  # 存储当前帧中已删除的跟踪对象。
        self.frame_id = 0   # frame_id: 当前处理的帧ID。
        self.args = args    # 跟踪参数，通常是一个包含各种配置的命名空间对象。
        self.det_thresh = args.track_thresh + 0.1  # 检测阈值，用于过滤检测结果。
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)  # 缓冲区大小，用于存储跟踪对象的帧数。根据帧率和跟踪缓冲参数计算。
        self.max_time_lost = self.buffer_size  # 最大丢失时间，超过这个时间的跟踪对象将被移除。
        self.kalman_filter = KalmanFilter()   # 卡尔曼滤波器，用于预测跟踪对象的下一帧位置。


    # 更新跟踪结果
    def update(self, output_results, img_info, img_size=(640, 640)):
        """接受检测头传递的结果数据"""
        # logger.info(f"update:{output_results}")     # [[4.8200e+02,  2.4262e+02,  6.1400e+02,  5.4700e+02,  9.9805e-01,9.6240e-01,  0.0000e+00]....] device='cuda:0'
        # logger.info(f"img_info:{img_info}")  # img_info:[720, 1280]  # 原图尺寸 720, 1280
        # logger.info(f"img_size:{img_size}")  #  img_size:(800, 1440)  #  模型的input大小
        # [[166, 226, 336, 697, 0.0, 0.870685875415802], [940, 221, 1169, 720, 0.0, 0.8511669635772705], [546, 274, 660, 521, 0.0, 0.8113520741462708], [392, 242, 512, 517, 0.0, 0.7862546443939209], [320, 346, 408, 566, 0.0, 0.7465958595275879], [750, 246, 859, 509, 0.0, 0.6751760244369507], [664, 268, 754, 504, 0.0, 0.5766164064407349], [132, 361, 187, 443, 26.0, 0.5731625556945801], [615, 205, 684, 446, 0.0, 0.5418628454208374], [808, 297, 873, 386, 26.0, 0.5153290629386902]]
        # 每次传入新的结果都需要更新帧ID:
        self.frame_id += 1

        # 初始化当前帧的跟踪状态:
        activated_starcks = []  # 存储当前帧中新激活的跟踪对象。
        refind_stracks = []     # 存储当前帧中重新激活的跟踪对象。
        lost_stracks = []       # 存储当前帧中丢失的跟踪对象。
        removed_stracks = []    # 存储当前帧中移除的跟踪对象。

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # print(img_h, img_w, "img_h, img_w")
        # print(img_size, "img_size")
        # print(scale, "scale")
        bboxes /= scale

        print(f"bboxes:{bboxes}")  # bboxes:[[  422.55       229.04999    548.55       504.44998 ].....
        # print(f"scores:{scores}")  # scores:[0.9605565  0.9561169  0.9507537  0.94548774 0.9430547  0.9289429.....]

        """step1: 处理检测结果 """
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []
        '''处理未确认的跟踪对象'''
        unconfirmed = []    # 存储当前帧中未确认的跟踪对象。
        tracked_stracks = []  # type: list[STrack]  # 存储当前帧中已确认的跟踪对象。
        for track in self.tracked_stracks:  # 遍历当前帧中所有活跃的跟踪对象。
            if not track.is_activated:  # 如果跟踪对象未激活，将其添加到 unconfirmed 列表。
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)   # 如果跟踪对象已激活，将其添加到 tracked_stracks 列表。

        ''' Step 2: 第一次匹配'''
        # 将已确认的跟踪对象和丢失的跟踪对象合并。
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # 使用卡尔曼滤波器预测所有跟踪对象的位置。
        STrack.multi_predict(strack_pool)
        # 计算已确认的跟踪对象和检测框之间的IOU距离。
        dists = matching.iou_distance(strack_pool, detections)
        #  如果不是MOT20数据集，融合得分。
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        # 使用线性分配算法进行匹配。
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # 更新匹配成功的跟踪对象
        for itracked, idet in matches: # 遍历匹配结果。
            track = strack_pool[itracked]   # 获取匹配的跟踪对象。
            det = detections[idet]  # 获取匹配的检测结果。
            if track.state == TrackState.Tracked:  # 如果跟踪对象是已激活状态，更新其状态。
                track.update(detections[idet], self.frame_id)   # 更新跟踪对象的状态。
                activated_starcks.append(track) # 将更新后的跟踪对象添加到 activated_starcks 列表。
            else:
                # 如果跟踪对象不是已激活状态，重新激活它。
                track.re_activate(det, self.frame_id, new_id=False) # : 重新激活跟踪对象。
                refind_stracks.append(track)    # 将重新激活的跟踪对象添加到 refind_stracks 列表。

        ''' Step 3: 第二次匹配'''
        # 如果有得分在0.1到 track_thresh 之间的检测框。
        if len(dets_second) > 0:
            '''将检测框和得分转换为 STrack 对象。'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            # 如果没有得分在0.1到 track_thresh 之间的检测框，detections_second 列表为空。
            detections_second = []
        """第二次匹配继续"""
        # 获取未匹配的已激活跟踪对象。
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 计算未匹配的已激活跟踪对象与低得分检测结果之间的IoU距离。
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 使用线性分配算法进行匹配。
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # 更新第二次匹配成功的跟踪对象
        for itracked, idet in matches:  # 遍历匹配结果。
            track = r_tracked_stracks[itracked] # 获取匹配的跟踪对象。
            det = detections_second[idet]   # 获取匹配的检测结果。
            #  如果跟踪对象是已激活状态，更新其状态。
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)    # 更新跟踪对象的状态。
                activated_starcks.append(track) # 将更新后的跟踪对象添加到 activated_starcks 列表。
            else:   # 如果跟踪对象不是已激活状态，重新激活它。
                track.re_activate(det, self.frame_id, new_id=False) # 重新激活跟踪对象。
                refind_stracks.append(track)    # 将重新激活的跟踪对象添加到 refind_stracks 列表。
        # 遍历未匹配的已激活跟踪对象。
        for it in u_track:
            track = r_tracked_stracks[it] # 获取未匹配的跟踪对象。
            if not track.state == TrackState.Lost:  # 如果跟踪对象不是丢失状态，标记为丢失。
                track.mark_lost()   #  标记跟踪对象为丢失。
                lost_stracks.append(track)  # 将丢失的跟踪对象添加到 lost_stracks 列表。

        '''处理未确认的跟踪对象'''
        detections = [detections[i] for i in u_detection]   # 获取未匹配的高得分检测结果。
        dists = matching.iou_distance(unconfirmed, detections)  # 计算未确认的跟踪对象与未匹配的高得分检测结果之间的IoU距离。
        # if not self.args.mot20: # 如果不是MOT20数据集，融合得分。
        dists = matching.fuse_score(dists, detections)  # 使用线性分配算法进行匹配。
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # 更新未确认的跟踪对象
        for itracked, idet in matches: # 遍历匹配结果。
            unconfirmed[itracked].update(detections[idet], self.frame_id)   # 更新未确认的跟踪对象。
            activated_starcks.append(unconfirmed[itracked]) # 将更新后的未确认跟踪对象添加到 activated_starcks 列表。
        # 遍历未匹配的未确认跟踪对象。
        for it in u_unconfirmed:
            track = unconfirmed[it] # 获取未匹配的未确认跟踪对象。
            track.mark_removed()    #  标记未确认的跟踪对象为移除。
            removed_stracks.append(track)   #  将移除的跟踪对象添加到 removed_stracks 列表。

        """ Step 4: 初始化新的跟踪对象"""
        for inew in u_detection:    # 遍历未匹配的高得分检测结果。
            track = detections[inew]    #  获取未匹配的检测结果。
            if track.score < self.det_thresh:   # 如果检测结果的得分低于 det_thresh，跳过。
                continue
            track.activate(self.kalman_filter, self.frame_id)   # 激活新的跟踪对象
            activated_starcks.append(track) # 将激活的跟踪对象添加到 activated_starcks 列表。
        """ Step 5: 更新丢失的跟踪对象"""
        # 遍历丢失的跟踪对象。
        for track in self.lost_stracks: # 如果跟踪对象丢失的时间超过 max_time_lost，标记为移除。
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()    # 标记跟踪对象为移除。
                removed_stracks.append(track)   # 将移除的跟踪对象添加到 removed_stracks 列表。

        # print('Ramained match {} s'.format(t4-t3))
        """last: 更新跟踪状态 """
        # 更新当前帧中活跃的跟踪对象列表。
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 合并当前帧中活跃的跟踪对象和新激活的跟踪对象。
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 合并当前帧中活跃的跟踪对象和重新激活的跟踪对象。
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 从丢失的跟踪对象中移除已激活的跟踪对象。
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 将丢失的跟踪对象添加到 lost_stracks 列表。
        self.lost_stracks.extend(lost_stracks)
        # 从丢失的跟踪对象中移除已移除的跟踪对象。
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 将移除的跟踪对象添加到 removed_stracks 列表。
        self.removed_stracks.extend(removed_stracks)
        # 移除重复的跟踪对象。
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 获取当前帧中活跃的跟踪对象。 返回当前帧中活跃的跟踪对象
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks



""" 将两个轨迹列表 tlista 和 tlistb 合并成一个新的列表，确保每个轨迹只出现一次。"""
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

"""从轨迹列表 tlista 中移除出现在轨迹列表 tlistb 中的所有轨迹。"""
def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


"""去除两个轨迹列表 stracksa 和 stracksb 中的重复轨迹。重复轨迹的判断依据是 IoU（Intersection over Union）距离小于 0.15。"""
def remove_duplicate_stracks(stracksa, stracksb):
    # IoU 距离：用于多目标跟踪中的轨迹匹配，衡量两个轨迹之间的相似度。成本矩阵中的值越小，表示两个轨迹越相似。
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb