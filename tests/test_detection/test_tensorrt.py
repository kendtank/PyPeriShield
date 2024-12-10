import torch
import torchvision.models as models
import torch_tensorrt
import time

def test_tensorrt_inference():
    # 1. 加载预训练的 ResNet-18 模型
    print("Loading model...")
    # 修改 Max Pooling 层，禁用 dilation
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval().cuda()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MaxPool2d) and module.dilation != 1:
            print(f"Modifying MaxPool2d layer: {name}")
            new_module = torch.nn.MaxPool2d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=1  # 禁用 dilation
            )
            setattr(model, name, new_module)


    # 2. 准备输入数据
    print("Preparing input data...")
    input_shape = (1, 3, 224, 224)  # 批量大小为 1，图像尺寸为 224x224
    input_tensor = torch.randn(input_shape).cuda()

    # 3. 使用 torch.jit.trace 将模型转换为静态图
    print("Tracing model...")
    traced_model = torch.jit.trace(model, input_tensor)

    # 4. 编译模型为 TensorRT 引擎
    print("Compiling model with TensorRT...")
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(input_shape)],
        enabled_precisions={torch.float32},  # 使用 FP32 精度
        truncate_long_and_double=True,
        # ir=torch_tensorrt.TorchScriptModule,  # 使用 TorchScript 编译
        debug=True
    )


    # 5. 运行推理
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

    # 6. 验证输出是否一致
    print("Comparing outputs...")
    max_diff = torch.max(torch.abs(torch_output - trt_output)).item()
    print(f"Maximum difference between PyTorch and TensorRT outputs: {max_diff:.6f}")

    if max_diff < 1e-5:
        print("Inference successful! Outputs are consistent.")
    else:
        print("Warning: Outputs differ significantly!")

if __name__ == "__main__":
    test_tensorrt_inference()




"""
import torch.quantization

# 启用 PTQ
model.fuse_model()  # 如果模型支持融合操作
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

# 编译量化后的模型
trt_model = torch_tensorrt.compile(
    traced_model,
    inputs=[torch_tensorrt.Input(input_shape)],
    enabled_precisions={torch.float16},  # 使用 FP16 精度
    truncate_long_and_double=True,
    ir=torch_tensorrt.TorchScriptModule,  # 使用 TorchScript 编译
    debug=True
)
"""