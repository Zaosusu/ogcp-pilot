import torch
import time

print("=" * 50)
print("GPU 诊断")
print("=" * 50)

print(f"\nPyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA 不可用，检查 PyTorch 安装")
    exit()

print(f"CUDA 版本: {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# 测试 GPU 计算速度
print("\n" + "=" * 50)
print("GPU 速度测试")
print("=" * 50)

device_gpu = torch.device("cuda")

# 预热
x = torch.randn(1000, 1000, device=device_gpu)
for _ in range(10):
    _ = torch.mm(x, x)
torch.cuda.synchronize()

# 正式测试
x = torch.randn(5000, 5000, device=device_gpu)
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    result = torch.mm(x, x)
    _ = result.sum()
torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"GPU 时间: {gpu_time:.3f}s")
print(f"✅ GPU 正常运行")
