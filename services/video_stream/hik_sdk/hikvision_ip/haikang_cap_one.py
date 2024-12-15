# coding=utf-8
"""
@Author: Kend
@Date: 2024/06/21
@Time: 14:44
@Description: 海康摄像头sdk采图的实现
@Modify:
@Contact: tankang0722@gmail.com
"""
import traceback

"""
海康采图：
采集的json按照cam_id:ip给出
现在是按照cam_id和时间命名存储，采集单张
"""

import time
import os
import platform
from HCNetSDK import *
from PlayCtrl import *
import threading
import datetime
import sys

sys.path.append('./lib/win')
# 保存的路径
CAMERA_SAVE_ROOT = "/haikui/3d_forml/temp_images"

WINDOWS_FLAG = True


# 获取当前系统环境
def GetPlatform():
    sysstr = platform.system()
    print('' + sysstr)
    if sysstr != "Windows":
        global WINDOWS_FLAG
        WINDOWS_FLAG = False


# 设置SDK初始化依赖库路径
def SetSDKInitCfg(Objdll):
    # 设置HCNetSDKCom组件库和SSL库加载路径
    # print(os.getcwd())
    if WINDOWS_FLAG:
        strPath = os.getcwd().encode('gbk') + b'\lib\win'
        print(strPath)
        sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
        sdk_ComPath.sPath = strPath
        Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
        Objdll.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'\libcrypto-1_1-x64.dll'))
        Objdll.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'\libssl-1_1-x64.dll'))
    else:
        strPath = os.getcwd().encode('utf-8')
        sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
        sdk_ComPath.sPath = strPath
        Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
        Objdll.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'/libcrypto.so.1.1'))
        Objdll.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'/libssl.so.1.1'))



class HaiKangCaptureThread(threading.Thread):
    def __init__(self, cam_id, ip, port, username, password):
        super(HaiKangCaptureThread, self).__init__()

        self.cam_id = cam_id
        self.kill = False
        # # 获取系统平台
        GetPlatform()
        # # 加载库,先加载依赖库
        if WINDOWS_FLAG:
            self.Objdll = ctypes.CDLL(r'./lib/win/HCNetSDK.dll')  # 加载网络库
        else:
            self.Objdll = cdll.LoadLibrary(r'./libhcnetsdk.so')
        self.SetSDKInitCfg()  # 设置组件库和SSL库加载路径
        # # 初始化DLL
        self.Objdll.NET_DVR_Init()
        # # 启用SDK写日志  一般不用
        # self.Objdll.NET_DVR_SetLogToFile(3, bytes('./SdkLog_Python/', encoding="utf-8"), False)
        # 登录设备
        (self.lUserId, device_info) = self.LoginDev(ip, port, username, password)
        # print(self.lUserId, ip)  # 10 10.134.124.125
        print("self.lUserId, device_info", self.lUserId, device_info)
        if self.lUserId < 0:
            err = self.Objdll.NET_DVR_GetLastError()
            print('Login device fail, error code is: %d' % err)
            # 释放资源
            self.Objdll.NET_DVR_Cleanup()

        # JPEG图像信息结构体
        self.jpeg_para = NET_DVR_JPEGPARA()
        self.jpeg_para.wPicSize = 9  # 图片尺寸: 9-HD1080P(1920*1080) 0xff-Auto(使用当前码流分辨率)
        self.jpeg_para.wPicQuality = 1  # 图片质量系数：0-最好，1-较好，2-一般

    def SetSDKInitCfg(self):
        # 设置HCNetSDKCom组件库和SSL库加载路径
        # print(os.getcwd())
        if WINDOWS_FLAG:
            strPath = os.getcwd().encode('gbk')
            sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
            sdk_ComPath.sPath = strPath
            self.Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
            self.Objdll.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'\lib\win\libcrypto-1_1-x64.dll'))
            self.Objdll.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'\lib\win\libssl-1_1-x64.dll'))
        else:
            strPath = os.getcwd().encode('utf-8')
            sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
            sdk_ComPath.sPath = strPath
            self.Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
            self.Objdll.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'/libcrypto.so.1.1'))
            self.Objdll.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'/libssl.so.1.1'))

    def LoginDev(self, ip, port, username, password):
        # 登录注册设备
        # print(1111,ip)
        DEV_IP = create_string_buffer(ip.encode('utf-8'))
        DEV_PORT = 8000
        DEV_USER_NAME = create_string_buffer(username.encode('utf-8'))
        DEV_PASSWORD = create_string_buffer(password.encode('utf-8'))
        device_info = NET_DVR_DEVICEINFO_V30()
        lUserId = self.Objdll.NET_DVR_Login_V30(DEV_IP, DEV_PORT, DEV_USER_NAME, DEV_PASSWORD, byref(device_info))
        return (lUserId, device_info)

    def capture(self):

        # 不分文件夹，所有图片放入同一个文件夹
        files = os.path.join(CAMERA_SAVE_ROOT, "test_0628")
        if not os.path.exists(files):
            os.makedirs(files)
        # 按照id存储，名为月日时分秒 + id
        jpg_path = os.path.join(files, "{}_{}.jpg".format(self.cam_id, datetime.datetime.now().strftime("%m%d%H%M%S")))
        print(f'save_file:{jpg_path}')

        ret = self.Objdll.NET_DVR_CaptureJPEGPicture(self.lUserId, 1, byref(self.jpeg_para),
                                                     c_char_p(jpg_path.encode()))
        if ret == 0:
            err = self.Objdll.NET_DVR_GetLastError()
            print('capture fail, error code is: %d' % err)
            return

    def run(self):
        num = 0
        # while True:
        #     try:
        #         self.capture()
        #     except Exception as e:
        #         print("errorrrrr", e)
        #     time.sleep(2)

         # 每两秒采一次图
        while not self.kill:
            try:
                t1 = datetime.datetime.now()
                self.capture()
                t2 = datetime.datetime.now()
                time_diff = t2 - t1
                time_diff = time_diff.seconds + time_diff.microseconds / 1000000
                if time_diff < 2:
                    time.sleep(2 - time_diff)
            except:
                print("退出采图", traceback.format_exc())
                self.Objdll.NET_DVR_Logout(self.lUserId)
                self.Objdll.NET_DVR_Cleanup()
                break




if __name__ == "__main__":

    t = HaiKangCaptureThread('test', '10.134.129.23', 8000, "admin", "lyric1234")  # hiv
    threading.Thread(target=t.run).start()
