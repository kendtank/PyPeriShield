import torch
import torchvision.models as models
import torch_tensorrt
import time


def test_tensorrt_compatibility():
    # 1. 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a GPU and the correct drivers installed.")
        return

    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch_tensorrt version: {torch_tensorrt.__version__}")

    # 2. 加载预训练的 ResNet-18 模型
    print("Loading model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval().cuda()

    # 3. 准备输入数据
    print("Preparing input py_db...")
    input_shape = (1, 3, 224, 224)  # 批量大小为 1，图像尺寸为 224x224
    input_tensor = torch.randn(input_shape).cuda()

    # 4. 使用 TorchScript 进行静态图编译
    print("Tracing model...")
    traced_model = torch.jit.trace(model, input_tensor)

    # 5. 编译模型为 TensorRT 引擎
    print("Compiling model with TensorRT...")
    try:
        trt_model = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input(input_shape)],
            enabled_precisions={torch.float32},  # 使用 FP32 精度
            truncate_long_and_double=True,
            ir=torch_tensorrt.TorchScriptModule,  # 使用 TorchScript 编译
            debug=True  # 启用调试模式
        )
        print("Model compiled successfully with TensorRT.")
    except Exception as e:
        print(f"Failed to compile model with TensorRT: {e}")
        return

    # 6. 运行推理
    print("Running inference with TensorRT...")
    with torch.no_grad():
        # 测试 PyTorch 模型的推理时间
        start_time = time.time()
        torch_output = model(input_tensor)
        torch_inference_time = time.time() - start_time
        print(f"PyTorch inference time: {torch_inference_time:.4f} seconds")

        # 测试 TensorRT 模型的推理时间
        start_time = time.time()
        trt_output = trt_model(input_tensor)
        trt_inference_time = time.time() - start_time
        print(f"TensorRT inference time: {trt_inference_time:.4f} seconds")

    # 7. 验证输出是否一致
    print("Comparing outputs...")
    max_diff = torch.max(torch.abs(torch_output - trt_output)).item()
    print(f"Maximum difference between PyTorch and TensorRT outputs: {max_diff:.6f}")

    if max_diff < 1e-5:
        print("Inference successful! Outputs are consistent.")
    else:
        print("Warning: Outputs differ significantly!")

if __name__ == "__main__":
    test_tensorrt_compatibility()


    # 我这个环境，还能升级torch_tensorrt吗？