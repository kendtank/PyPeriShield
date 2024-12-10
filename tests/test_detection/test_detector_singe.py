"""
@Time    : 2024/11/23 下午11:42
@Author  : Kend
@FileName: test-.py
@Software: PyCharm
@modifier:
"""
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)   # 将project_root目录更改为项目的根目录。
# sys.path.append(project_root) # 还是找不到就把项目路劲加到python解释器中
import time
import cv2
import numpy as np
from core.detection.detector import Detector

def main():
    # 模拟多个图像
    images = [cv2.imread("tests/frame_0000.jpg") for _ in range(100)]
    # images = [np.random.rand(640, 640, 3) for _ in range(5)]  # 生成 5 张随机图像

    # 创建 Detector 实例
    detector = Detector(
        ckpt_path=r"weights/yolo11n.pt",  # 替换为实际的模型路径
        input_size=(640, 640),
        half=False,
        detector='yolo11',  # 或 'yolov5'
        num_workers=4
    )

    # for img in images:
        # 执行推理任务
        # detector.async_inference_wait(img)
        # detector.inference(img)
    detector.work(images)
    # return results

    # 打印推理结果
    # for i, (tensor_board_data, ratio) in enumerate(results):
    #     print(f"图像 {i+1} 的推理结果:")
        # print(f"  tensor_board_data: {tensor_board_data}")
        # print(f"  ratio: {ratio}")


if __name__ == "__main__":
    time1 = time.time()
    main()
    print("time cost:", time.time() - time1)

    """
    test 100 images 27 labels
    单进程批处理   time cost: 8.518536567687988 s  8.58 s  4 num-work 9.64s  num_work16 12 s 越高越慢
    单进程逐帧处理 time cost: 8.498108863830566 s  8.62 s  4 num-work 8.58s  num_work16 8.7 s 稳定
    不采用线程池   time cost: 8.756960535049438 s  8.67 s  4 num-work 8.57s  num_work16 8.6 s 
    总结:
    单进程批处理：在批处理模式下，使用线程池（特别是当 num_workers 较大时）反而导致了性能下降。这可能是因为线程池的管理开销（如任务调度、上下文切换等）超过了多线程带来的并行加速效果。特别是在 I/O 密集型任务中，过多的线程可能会导致竞争资源（如 CPU、内存带宽），从而降低整体性能。
    单进程逐帧处理：逐帧处理时，线程池的表现相对稳定，且随着 num_workers 的增加，时间成本略有上升，但幅度较小。这表明逐帧处理的任务可能更适合使用少量线程来并行化，而不需要过多的线程。
    不采用线程池：不使用线程池的情况下，性能表现较为稳定，且不受 num_workers 设置的影响。这进一步验证了线程池的管理开销在某些场景下可能会抵消多线程的优势。
    如果没有线程池，性能与逐帧处理类似。这意味着创建和管理线程的开销不会有利于单图像顺序任务。
    """