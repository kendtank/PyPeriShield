import os

import torch
import torch.onnx


# 加载 PyTorch 模型
def load_model(model_path):
    model = torch.load(model_path, map_location='cuda')
    model.eval()
    return model


# 导出为 ONNX 格式
def torch2onnx(model_path, onnx_path):
    model = load_model(model_path)
    dummy_input = torch.randn(1, 3, 32, 448, device='cuda')  # 根据您的模型输入尺寸调整
    input_names = ['input']
    output_names = ['output']

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"}}  # 如果需要动态推理
    )
    print('->> 模型转换成功！')


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(project_root)
    # 调用函数进行转换
    model = "weights/yolo11n.pt"
    torch2onnx('model', 'yolo11n.onnx')
