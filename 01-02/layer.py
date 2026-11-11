import numpy as np

x = np.array([0.5, 0.8, 0.2])           # input: 3 numbers
W = np.random.randn(2, 3)           # weights: what shape?
b = np.array([0.1, 0.1])                # bias: one per output

output = np.maximum(0, W @ x + b)        # relu(W @ x + b), written with np.maximum

print("W shape:", W.shape)
print("output:", output)