import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.__version__)

import torchaudio
print(torchaudio.__version__)
import torchvision
print(torchvision.__version__)

import tensorrt
print(tensorrt.__version__)  # 8.6.0

import torch_tensorrt
print(torch_tensorrt.__version__)  # 2.2.0

