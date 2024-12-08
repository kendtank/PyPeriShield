#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/12/7
@Time: 10:58
@Description: tensorrt - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""

import torch
import torch_tensorrt

print(torch_tensorrt.__version__)

engine_file_path = "/home/lyh/work/depoly/PyPeriShield/weights/yolo11s.pt"

# 加载你的PyTorch模型
model = torch.load(engine_file_path)

# 将模型转换为TensorRT引擎
trt_model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 3, 224, 224))])

# 运行TensorRT引擎
input_data = torch.randn(1, 3, 224, 224)
output = trt_model(input_data)



