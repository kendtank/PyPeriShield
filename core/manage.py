#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:44
@Description: demo - manage的一个demo实现
@Modify:
@Contact: tankang0722@gmail.com
"""

import os

from core.tracking.tracker import BYTETracker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(project_root)  # /home/lyh/work/depoly/PyPeriShield-feature
os.chdir(project_root)
from core.detection.detector import Detector
from services.message_queue.mq import CameraMQ, Consumer
from services.video_stream.rtsp_stream import RTSPCamera
from tools.logger import logger
from tools.load_yaml import load_yaml_file


# self.arg = load_yaml_file("config/config.yaml")
# t = tuple(self.arg['algorithm_para']['input_size'])
# print(t, type(t))


class CameraShield:
    """ 摄像头管理类 以摄像头为单位 - 采图-推理-事件判定-回放 """

    def __init__(self, came_ip):
        self.came_ip = came_ip
        self.arg = load_yaml_file("../PyPeriShield/config/config.yaml")
        self.camera_id = None
        self.camera_rtsp_url = None

    # TODO 获取摄像头参数， 区域，判定线， 区域入侵区域
    def get_camera_arg(self):
        pass

    def run(self):
        # TODO 获取摄像头参数， 区域，判定线， 区域入侵区域
        self.get_camera_arg()
        # 初始化配置参数
        # 初始化检测头
        predictor = Detector(
            self.arg['algorithm_para']['ckpt_path'],
            self.arg['algorithm_para']['confidence_threshold'],
            self.arg['algorithm_para']['iou_threshold'],
            tuple(self.arg['algorithm_para']['input_size']),
            self.arg['algorithm_para']['half'],
            self.arg['algorithm_para']['iou_type'],
            self.arg['algorithm_para']['num_workers'],
            self.arg['algorithm_para']['model_type'],
        )
        # 初始化 MQ 生产者
        mq_producer = CameraMQ(camera_id=self.camera_id)
        # 初始化图像采集
        camera_stream = RTSPCamera(
            camera_id=self.camera_id,
            rtsp_url=self.camera_rtsp_url,
            save_dir=self.arg['cameras']['save_dir'],
            mq_producer=mq_producer
        )
        camera_stream.run()

        # 推理+跟踪

        # 轨迹后处理

        # 事件判定

        # 回放

        # 推送

    def inference_and_tracker(self):
        # 初始化跟踪器
        tracker = BYTETracker(args, frame_rate=self.arg)  # 根据帧率决定缓存区

    def playback(self):
        pass

    # @staticmethod
    # def capture(camera_id, rtsp_url, save_dir, mq_producer):
    #     return


#
""" 视频推理的演示方法 demo """


def imageflow_demo(predictor):
    """
    Args:
        predictor: # 检测头
        args: # 参数类
    Returns:
    """

    # 检测的图像标记帧数号
    frame_id = 0
    results = []
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            # 推理帧图像，
            # cv2.imshow("image", frame)
            # cv2.waitKey(0)
            outputs, img_info = predictor.predict(frame, timer)
            # print("yolo11:", outputs)
            if outputs is not None:
                # logger.info(f"检测头推理的结果：{outputs[0]}")   # 二维张量， 每一行都是7个数
                online_targets = tracker.update(outputs, [img_info['height'], img_info['width']],
                                                img_size=args.input_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                # print('online_targets', online_targets)
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    # print("tlwh[2] / tlwh[3]", tlwh[2] / tlwh[3])
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        # if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.stop()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.stop()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


"""获取路径中的所有图像"""


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


"""将多目标跟踪结果写入文件的函数。
这个函数接受一个文件名和结果数据，然后按照指定的格式将结果写入文件。
"""


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def video2images(video_path, output_folder, image_format='jpg'):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件 {video_path}")
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            # 构建图像文件名
            image_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.{image_format}")
            # 保存图像文件
            cv2.imwrite(image_filename, frame)
        frame_count += 1
    # 释放视频捕获对象
    cap.release()
    print(f"转换完成，共生成 {frame_count} 张图像。")


def main(args):
    logger.info(f"args.model_path:{args.model_path}")
    predictor = PredictorYolo11(model_path=args.model_path, input_size=args.input_size)
    current_time = time.localtime()
    imageflow_demo(predictor, args.save_result, current_time, args)


class Args:
    # config
    demo = "video"
    path = r"D:\kend\myPython\Hk_Tracker\data\videos\palace.mp4"
    # path = r"D:\kend\work\Hk_Tracker\py_db\dataset\test_images"
    save_result = r"D:\kend\myPython\Hk_Tracker\visualization\vis_folder\demo_output"
    fps = 30
    counts = 30
    # model
    model_path = r"D:\kend\myPython\Hk_Tracker\detector_head\yolo11\yolo11s.onnx"
    input_size = (640, 640)
    fp16 = False  # cpu no half
    # tracker
    track_thresh = 0.65  # 追踪置信度阈值,低于此值则二次匹配追踪
    track_buffer = 60  # 缓存帧数，决定最大丢失时间 2秒
    match_thresh = 0.8  # 代价匹配阈值，太低容易目标id被重合的id带走
    min_box_area = 16
    aspect_ratio_thresh = 3


if __name__ == "__main__":
    # args = make_parser().parse_args()
    args = Args()
    main(args)
    # python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py  --fp16 --fuse --save_result

