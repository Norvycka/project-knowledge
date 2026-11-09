import torch

for step in range(10):
    x = torch.tensor([1.0, 2.0, float(5 - step)])
    loss = torch.log(x).mean()

    if torch.isnan(loss) or torch.isinf(loss):
        breakpoint()

    print(step, loss.item())
