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
from predictor_yolo11 import PredictorYolo11
print("torch_tensorrt.version:", torch_tensorrt.__version__)

torch_file_path = "/home/lyh/work/depoly/PyPeriShield-feature/weights/yolo11n.pt"

# 加载你的PyTorch模型
model_class = PredictorYolo11(torch_file_path, input_size=(640, 640), fp16=False)
print(type(model_class.model))

model = model_class.model
# 将模型转换为TensorRT引擎
trt_model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 3, 224, 224))])

# 运行TensorRT引擎
input_data = torch.randn(1, 3, 224, 224)
output = trt_model(input_data)



