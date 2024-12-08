#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2023/12/03
@Time: 15:19
@Description: post_process_image - 推理张量的后处理
@Modify:
@Contact: tankang0722@gmail.com
"""

import torch
import torchvision
from mmdet.core import bbox_overlaps

"""
如果一个预测框的置信度是0.9，表示模型认为这个框内很可能包含一个对象。然后，模型预测这个对象属于 "猫" 类别的概率是0.8，属于 "狗" 类别的概率是0.2。
那么，这个预测框内对象属于 "猫" 类别的类别置信度就是 0.9 * 0.8 = 0.72，属于 "狗" 类别的类别置信度就是 0.9 * 0.2 = 0.18。
总的来说，置信度是对预测框内是否存在对象的总体评价，而类别置信度则是对预测框内对象属于某个具体类别的评价。
"""


"""推理张量的后处理, 支持图像的批处理，取决于prediction的batch_size"""
def postprocess(prediction, num_classes, conf_thre=0.5, nms_thre=0.4, iou='iou'):
    """
    Args:
        prediction: 支持模型的预测输出为xywh，yolo的检测头输出为形状为 (batch_size, num_boxes,
        5 + num_classes)。其中，前4个值表示边界框的中心点坐标和宽度高度，第5个值表示对象置信度，剩下的值表示各个类别的概率。
        num_classes: 类别数量。
        conf_thre: 置信度阈值，用于过滤掉低置信度的预测框，默认值为 0.5。
        nms_thre: 非极大值抑制的阈值，默认值为 0.4。
        iou: 是否启用ciou预测，默认为 iou
    Returns:
        # example: 这里的矩形框没有放缩为原图
            output = [
                torch.tensor([
                    [10.0, 20.0, 50.0, 60.0, 0.9, 0.8, 1.0],  # 第一个图像的第一个检测框 + 置信度+ 类别置信度+ 类别索引
                    [100.0, 150.0, 200.0, 250.0, 0.85, 0.75, 2.0]  # 第一个图像的第二个检测框
                ]),
                torch.tensor([
                    [15.0, 25.0, 55.0, 65.0, 0.88, 0.82, 3.0],  # 第二个图像的第一个检测框
                    [110.0, 160.0, 210.0, 260.0, 0.8, 0.7, 1.0]  # 第二个图像的第二个检测框
                ])
            ]
            # object_conf: 对象置信度。class_conf: 类别置信度。class_pred: 类别索引（整数）。
    """

    # print("prediction.shape", prediction.shape)  # # prediction.shape torch.Size([1, 25200, 6])  640*640+box+class+score
    box_corner = prediction.new(prediction.shape)  # 创建一个与 prediction 形状相同的张量 box_corner
    # # 将边界框的中心点坐标和宽度高度转换为边界框的四个角点坐标  xywh -> xyxy
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # 左上角的 x 坐标。
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # 左上角的 y 坐标。
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # 右下角的 x 坐标。
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # 右下角的 y 坐标。
    prediction[:, :, :4] = box_corner[:, :, :4]    # 将转换后的边界框坐标赋值回 prediction 中 [x_min, y_min, x_max, y_max]

    # print(len(prediction))  # 1  代表一张图， 三维数组只返回最外层的列表长度，其实就是batch_size
    # 创建一个长度为 batch_size 的列表 output，用于存储每张图的最终检测结果,
    output = [None for _ in range(len(prediction))]
    # 处理每张图像的预测结果：
    for i, image_pred in enumerate(prediction):
        # print("image_pred", image_pred)  # (batch_size, num_features) 批量大小 + 特征数量。
        # 如果图像没有预测框，则跳过该图像。
        if not image_pred.size(0):
            continue
        # 对于每一个目标框获取最高置信度的类别的索引。和置信度：
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # print(class_conf, class_pred)
        # 计算每个预测框的置信度与类别置信度的乘积，和 置信度阈值 conf_thre 进行过滤。
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()  # 布尔数组
        # 将边界框坐标、对象置信度、类别置信度和类别索引组合成一个新的张量 detections。
        # NOTE: 是这里决定了输出是七个特征值 边界框的坐标和置信度，class_conf 类别置信度，class_pred.float() 包含类别索引。因此，总共是7个特征值。
        # 而原生的yolov5是只保留了置信度
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        # 然后使用 conf_mask 过滤掉低置信度的预测框。
        detections = detections[conf_mask]
        # 如果预测框都被阈值过滤掉了，则跳过当前图像。
        if not detections.size(0):
            continue
        # NOTE NMS 使用 torchvision.ops.batched_nms 进行非极大值抑制，去除重叠的预测框。可以考虑使用其他方法，如 ciou Soft-NMS 等
        # nms_out_index 是经过 NMS 后保留下来的预测框的索引。
        if iou in ['ciou', 'giou']:
            # Use gIoU for NMS
            from mmcv.ops import bbox_overlaps
            # 获取所有类别的唯一值
            unique_labels = detections[:, 6].unique()
            class_detections_list = []
            for c in unique_labels:
                # 获取当前类别的所有预测框数据
                class_detections = detections[detections[:, 6] == c]
                scores = class_detections[:, 4] * class_detections[:, 5]
                boxes = class_detections[:, :4]
                keep_indices = nms_by_iou(boxes, scores, nms_thre, iou_type='ciou') if iou == 'ciou' else (
                    nms_by_iou(boxes, scores, nms_thre, iou_type='giou'))
                class_detections_list.append(class_detections[keep_indices])
            # 进行非极大值抑制（NMS）后，将所有类别的检测结果合并成一个张量 detections。
            detections = torch.cat(class_detections_list, dim=0) if class_detections_list else detections
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
            # 将经过 NMS 后的预测框赋值给 detections。
            detections = detections[nms_out_index]
        # 将经过 NMS 处理后的检测结果存储到 output 列表中。如果 output[i] 已经有内容，则将其与新的检测结果合并。
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    # 返回每个图像的最终检测结果。NOTE: output 是一个列表，每个元素是一个张量，表示一张图像的检测结果。且没有放缩为原图的尺寸
    return output



def nms_by_iou(boxes, scores, iou_threshold, iou_type='ciou'):
    """
    Perform non-maximum suppression using CIoU.

    Args:
        boxes: A tensor of shape (N, 4) containing the bounding boxes in xyxy format.
        scores: A tensor of shape (N,) containing the scores of the detections.
        iou_threshold: The threshold for non-maximum suppression.

    Returns:
        A tensor of indices of the kept detections.
    """
    if len(scores) == 0:
        return torch.tensor([], dtype=torch.int64)

    # 按照置信度分数从高到低排序
    order = torch.argsort(scores, descending=True)
    keep = []

    while len(order) > 0:
        # 选择当前置信度最高的边界框
        idx = order[0]
        keep.append(idx)
        # 计算当前框与其他框之间的 CIoU
        # ciou_matrix = bbox_overlaps(boxes[idx].unsqueeze(0), boxes[order[1:]], mode='ciou')
        if iou_type == 'giou':
            # print('giou')
            iou_matrix = bbox_overlaps(boxes[idx].unsqueeze(0), boxes[order[1:]], mode='giou')
        else:
            # print('ciou')
            iou_matrix = c_iou(boxes[idx].unsqueeze(0), boxes[order[1:]])
        ciou = iou_matrix[0]
        # 找出 CIoU 小于阈值的框
        mask = ciou < iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64)



"""手动实现的c-iou"""
def c_iou(box1, box2):
    """
    Calculate the Complete Intersection over Union (CIoU) between two sets of bounding boxes.

    Args:
        box1: A tensor of shape (n, 4) representing the first set of bounding boxes in xyxy format.
        box2: A tensor of shape (m, 4) representing the second set of bounding boxes in xyxy format.

    Returns:
        A tensor of shape (n, m) containing the CIoU values.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area

    iou = inter_area / union_area

    enclose_x1 = torch.min(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    enclose_y1 = torch.min(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    enclose_x2 = torch.max(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    enclose_y2 = torch.max(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    center_distance_sq = (b1_x1.unsqueeze(1) - b2_x1.unsqueeze(0)) ** 2 + (b1_y1.unsqueeze(1) - b2_y1.unsqueeze(0)) ** 2

    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + 1e-7)) - torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + 1e-7)), 2)
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (center_distance_sq / enclose_diag_sq + v * alpha)
    return ciou