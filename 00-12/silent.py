import torch

def debug_print(name, t):
	print(f'name,shape={t.shape}, min={t.min().item()}, max={t.max().item()}, mean={t.mean().item()}, isnan={t.isnan().any().item()}, isinf={t.isinf().any().item()}')

x = torch.tensor([1.0, 2.0, 0.0])
y = torch.log(x)
print(y)
print(y.mean())
print(y * 0)

debug_print("x", x)
debug_print("y", y)
