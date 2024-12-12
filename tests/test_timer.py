# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/8 下午3:31
@Author  : Kend
@FileName: test_timer.py
@Software: PyCharm
@modifier:
"""
import threading
import time
from tools import timer

def test_timer():
    t = timer()
    for i in range(5):
        t.start()
        time.sleep(i+1)
        duration = t.stop(average=False)
        print(f"线程{threading.get_ident()}单次耗时：", duration)  # 1 2 3 4 5
    print(f"总耗时平均：", t.average_time)  # (1+2+3+4+5)/5=3
    print(f"总耗时为：{t.total_time}")  # 15
    t.clear()


def test_timer_thread():
    for i in range(5):
        threading.Thread(target=test_timer).start()
        time.sleep(1.2)



if __name__ == '__main__':

    test_timer_thread()


    # # 启动计时器
    # timer.start()
    # # 模拟一些耗时操作
    # time.sleep(2)
    # # 停止计时器并获取平均时间
    # duration = timer.stop()
    # print(f"Duration: {duration} seconds")
    # # 再次启动计时器
    # timer.start()
    # # 模拟另一段耗时操作
    # time.sleep(3)
    # # 停止计时器并获取单次时间差
    # duration = timer.stop(average=False)
    # print(f"Single duration: {duration} seconds")
    # # 打印平均时间
    # print(f"Average duration: {timer.average_time} seconds")
    # # 清除计时器
    # timer.clear()
