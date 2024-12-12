# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/8 下午1:16
@Author  : Kend
@FileName: test_detector_multi.py
@Software: PyCharm
@modifier:
"""
import numpy as np
from core.detection.detector import Detector


def worker_process(ckpt_path, input_size, half, detector, num_workers, image, result_queue):
    """
    Worker process to perform inference using Detector singleton.

    Args:
        ckpt_path: Path to the model checkpoint.
        input_size: Input size for the model.
        half: Use FP16 if True.
        detector: Detector type (e.g., yolo11, yolov5).
        num_workers: Number of threads for async inference.
        image: Image for inference.
        result_queue: Multiprocessing queue to store the result.
    """
    detector_instance = Detector(ckpt_path, input_size, half, num_workers, detector)
    future = detector_instance.async_inference(image)
    result_queue.put(future.result())  # Send result back to the main process


def main():
    from multiprocessing import Manager, Process

    # Prepare shared py_db
    manager = Manager()
    result_queue = manager.Queue()

    # Simulate an image for inference
    image = np.random.rand(640, 640, 3)
    num_workers = 4  # Adjust based on your concurrency needs

    # Start multiple processes
    processes = []
    for i in range(4):  # Example: 4 processes
        process = Process(
            target=worker_process,
            args=(r"D:\kend\WorkProject\PyPeriShield\weights\yolo11n.pt", (640, 640), False, "yolo11", num_workers,
                  image, result_queue)
        )
        processes.append(process)
        process.start()

    # Collect results from all processes
    all_results = []
    for process in processes:
        process.join()
        all_results.append(result_queue.get())

    print("Final Results:", all_results)



if __name__ == '__main__':
    main()