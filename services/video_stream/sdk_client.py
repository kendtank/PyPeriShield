# 按照sql中的数据表，区分dahua还是海康， 再调用采图，
# 测试过主码流在百兆宽带下， 采集速度是2帧左右，千兆在10帧左右，图像大小在2-300kb 相当可以了
# 考虑到sdk的稳定性比rtsp好太多，唯一的缺点就是帧率上不如。有同事测试过24小时稳定性 10帧