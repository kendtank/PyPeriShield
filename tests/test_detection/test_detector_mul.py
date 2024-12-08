# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/7 下午11:55
@Author  : Kend
@FileName: test_detector_mul.py
@Software: PyCharm
@modifier:
"""
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Manager
from core.detection.detector import Detector
import time
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)


def worker_process(images, result_queue, ckpt_path, input_size, half, num_workers, detector):
    """
    工作进程函数，用于执行推理任务。

    Args:
        images (list[np.ndarray]): 要处理的图像列表。
        result_queue (Queue): 用于存储结果的多进程队列。
        ckpt_path (str): 模型检查点的路径。
        input_size (tuple): 模型输入的尺寸。
        half (bool): 是否使用 FP16 精度。
        num_workers (int): 线程池中的工作线程数量。
        detector (str): 检测器类型（例如 'yolo11', 'yolov5'）。
    """
    # 创建 Detector 实例
    detector_instance = Detector(
        ckpt_path=ckpt_path,
        input_size=input_size,
        half=half,
        num_workers=num_workers,
        detector=detector
    )
    results = []
    # 执行推理任务
    for img in images:
        results.append(detector_instance.inference(img))
    # results = detector_instance.work(images)

    # # 将结果放入队列中
    result_queue.put(results)

    # # 关闭线程池
    # detector_instance.shutdown()

def multi_process_multi_thread_test(images, num_processes, num_workers, ckpt_path, input_size, half, detector):
    """
    多进程 + 多线程测试函数。

    Args:
        images (list[np.ndarray]): 要处理的图像列表。
        num_processes (int): 进程数量。
        num_workers (int): 每个进程中的线程池工作线程数量。
        ckpt_path (str): 模型检查点的路径。
        input_size (tuple): 模型输入的尺寸。
        half (bool): 是否使用 FP16 精度。
        detector (str): 检测器类型（例如 'yolo11', 'yolov5'）。
    """

    # 使用 Manager 创建共享的 Queue
    manager = Manager()
    result_queue = manager.Queue()

    # 将图像列表均匀分配给每个进程
    chunk_size = len(images) // num_processes
    image_chunks = [images[i * chunk_size : (i + 1) * chunk_size] for i in range(num_processes)]
    if len(images) % num_processes != 0:
        image_chunks[-1].extend(images[num_processes * chunk_size:])  # 处理剩余的图像

    # 记录开始时间
    start_time = time.time()

    # 启动多个进程
    processes = []
    for i in range(num_processes):
        process = mp.Process(
            target=worker_process,
            args=(
                image_chunks[i],
                result_queue,
                ckpt_path,
                input_size,
                half,
                num_workers,
                detector
            )
        )
        processes.append(process)
        process.start()

    # 收集所有进程的结果
    all_results = []
    for process in processes:
        process.join()  # 等待子进程完成
        results = result_queue.get()  # 从队列中获取结果
        all_results.extend(results)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 打印结果
    print(f"总图像数量: {len(images)}")
    print(f"进程数量: {num_processes}")
    print(f"每个进程中的线程数量: {num_workers}")
    print(f"时间成本: {elapsed_time:.4f} 秒")

    return all_results


if __name__ == "__main__":
    import sys

    # 模拟 100 张图像
    images = [cv2.imread("tests/frame_0000.jpg") for _ in range(200)]

    # 测试参数
    num_processes = 8  # 进程数量
    num_workers = 1    # 每个进程中的线程池数量
    ckpt_path = r"weights/yolo11n.pt"  # 替换为实际的模型路径
    input_size = (640, 640)
    half = False
    detector = 'yolo11'  # 或 'yolov5'

    # 运行多进程 + 多线程测试
    results = multi_process_multi_thread_test(
        images=images,
        num_processes=num_processes,
        num_workers=num_workers,
        ckpt_path=ckpt_path,
        input_size=input_size,
        half=half,
        detector=detector
    )

    """
    多进程测试：200 pics
    批处理: 4+2 -> 11.5s  4+4-> 12.2s  8+2-> 13.6s  8+4: 14.6s
    逐帧处理 4+2-> 11.2s   4+4-> 11.1s  8+2 13.4s   8+4: 13.1s
    不使用线程池 4 -> 11.3s  8: -> 12.7s
    
    """