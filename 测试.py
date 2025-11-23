import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本(Pytorch 编译时):", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU 设备名称:", torch.cuda.get_device_name(0))