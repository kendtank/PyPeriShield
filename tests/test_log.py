#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/7
@Time: 09:47
@Description: test_log - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""


from utils.logger.log_factory import logger
import threading
import multiprocessing

def worker():
    logger.debug(f"Thread {threading.current_thread().name} is running")
    logger.info(f"Thread {threading.current_thread().name} is running")
    logger.warning(f"Thread {threading.current_thread().name} is running")
    logger.error(f"Thread {threading.current_thread().name} is running")
    logger.critical(f"Thread {threading.current_thread().name} is running")

def process_worker():
    logger.debug(f"Process {multiprocessing.current_process().name} is running")
    logger.info(f"Process {multiprocessing.current_process().name} is running")
    logger.warning(f"Process {multiprocessing.current_process().name} is running")
    logger.error(f"Process {multiprocessing.current_process().name} is running")
    logger.critical(f"Process {multiprocessing.current_process().name} is running")


if __name__ == "__main__":
    # 多线程测试
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # 多进程测试
    processes = [multiprocessing.Process(target=process_worker) for _ in range(10)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    # try:
    #     # 可能会引发异常的代码
    #     result = 10 / 0
    # except Exception as e:
    #     # logger.exception("An error occurred")
    #     # logger.error("An error occurred", e)

