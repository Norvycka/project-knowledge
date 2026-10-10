import torch
import time

def benchmark(size):
    a = torch.randn(size, size)
    b = torch.randn(size,size)
    
    start = time.perf_counter()
    c = a @ b
    cpu_time = time.perf_counter() - start
    
    
    a_gpu = a.to("cuda")
    b_gpu = b.to("cuda")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start
    
    print(f"{size}x{size}: CPU {cpu_time:.5f}s | GPU {gpu_time:.5f}s | speedup {cpu_time/gpu_time:.1f}x")
    
w = torch.randn(10, 10).to("cuda")
g = w @ w
for size in [10, 10, 100, 1000, 4000]:
    benchmark(size)