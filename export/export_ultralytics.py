from ultralytics import YOLO
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化 CUDA
import numpy as np
import cv2


def convert_model_to_engine(model_path, output_dir='.', format='engine', device=0, half=True, simplify=True, workspace=4.0, dynamic=True, imgsz=640, batch=1):
    """
    将 PyTorch 模型 (.pt) 转换为 TensorRT 引擎文件 (.engine)。

    参数:
        model_path (str): 输入的 .pt 模型文件路径。
        output_dir (str): 输出目录，默认为当前目录。
        format (str): 导出的目标格式，默认为 'engine'。
        device (int): 使用的 GPU 设备编号，默认为 0。
        half (bool): 是否启用 FP16 精度推理，默认为 True。
        simplify (bool): 是否简化 ONNX 模型图，默认为 True。
        workspace (float): TensorRT 优化的最大工作空间大小（单位：GiB），默认为 4.0。
        dynamic (bool): 是否支持动态输入尺寸，默认为 True。
        imgsz (int or tuple): 模型输入的期望图像尺寸，默认为 640x640。可以是整数表示正方形图像，也可以是元组 (height, width) 表示具体尺寸。
        batch (int): 最大批处理大小，默认为 1。当 dynamic=True 时必须指定。

    返回:
        str: 生成的 TensorRT 引擎文件路径。
    """

    # 加载模型
    model = YOLO(model_path)
    # 设置输出文件名
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.engine")


    # 定义最小、优化和最大输入形状
    min_shape = (1, 3, 32, 32)  # 最小输入形状
    opt_shape = (1, 3, imgsz, imgsz)  # 优化输入形状
    max_shape = (batch, 3, 1024, 1024)  # 最大输入形状

    # 创建优化配置文件
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    profile = builder.create_optimization_profile()

    # 将 Python 元组转换为 tensorrt.tensorrt.Dims
    min_dims = trt.Dims(min_shape)
    opt_dims = trt.Dims(opt_shape)
    max_dims = trt.Dims(max_shape)

    # 设置输入形状
    profile.set_shape('images', min=min_dims, opt=opt_dims, max=max_dims)

    # 导出模型为 TensorRT 引擎
    model.export(
        format=format,
        device=device,
        half=half,
        simplify=simplify,
        workspace=workspace,
        dynamic=dynamic,
        imgsz=imgsz,
        batch=batch,  # 显式设置最大批处理大小
        # profile=profile  # 传递优化配置文件
        #  TODO  task='detect',  # 显式指定任务类型 不影响但是会造成warning
    )

    print(f"Model converted successfully and saved to: {output_file}")
    return output_file



def validate_engine(engine_file, input_image_path='test.jpg'):
    """
    验证生成的 TensorRT 引擎文件是否正常工作。

    参数:
        engine_file (str): 输入的 TensorRT 引擎文件路径。
        input_image_path (str): 用于验证的输入图像路径，默认为 'test.jpg'。

    返回:
        bool: 如果验证成功返回 True，否则返回 False。
    """
    try:
        # 加载 TensorRT 引擎
        with open(engine_file, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        # 创建执行上下文
        context = engine.create_execution_context()

        # 打印引擎信息
        print(f"Engine name: {engine.name}")
        print(f"Number of bindings: {engine.num_bindings}")
        print(f"Max batch size: {engine.max_batch_size}")

        # 加载输入图像并预处理
        image = cv2.imread(input_image_path)
        if image is None:
            raise FileNotFoundError(f"Input image not found: {input_image_path}")

        # 假设输入图像需要调整为 640x640，您可以根据实际情况调整
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0  # 归一化
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # 添加批次维度

        # 分配输入输出缓冲区
        inputs = [np.ascontiguousarray(image)]
        outputs = [np.empty((1, 8400, 85), dtype=np.float32)]  # 假设输出形状为 (1, 8400, 85)，您可以根据实际情况调整

        # 获取输入输出绑定索引
        input_binding_index = engine.get_binding_index('images')
        output_binding_index = engine.get_binding_index('output0')
        # 创建 CUDA 流
        stream = cuda.Stream()
        # 执行推理
        context.execute_async_v2(
            bindings=[inputs[0].ctypes.data, outputs[0].ctypes.data],
            stream_handle=stream.handle
        )
        stream.synchronize()

        # 打印推理结果
        print("Inference completed successfully!")
        print("Output shape:", outputs[0].shape)

        return True

    except Exception as e:
        print(f"Validation failed: {e}")
        return False


if __name__ == "__main__":

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    print(project_root)
    # 指定模型路径和输出目录
    model_path = 'weights/yolo11n.pt'
    output_dir = '/home/lyh/work/depoly/PyPeriShield-feature/weights'  # 需要绝对路径
    # 转换模型为 TensorRT 引擎
    engine_file = convert_model_to_engine(
        model_path=model_path,
        output_dir=output_dir,
        half=True,
        simplify=True,
        workspace=4.0,
        dynamic=True,
        imgsz=640,
        batch=1  # 显式设置最大批处理大小
    )

    # 验证生成的 TensorRT 引擎
    if validate_engine(engine_file, input_image_path='/home/lyh/work/depoly/PyPeriShield-feature/tests/frame_0000.jpg'):
        print("Engine validation passed!")
    else:
        print("Engine validation failed.")