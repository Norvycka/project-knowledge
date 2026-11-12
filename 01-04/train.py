qimport numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()   # random start: the model knows nothing
lr = 0.01

for epoch in range(1000):
    pred = w * x + b                 # 1. predict
    error = pred - y                 # 2. how wrong, for each point
    loss = np.mean(error ** 2)       # 3. one number: average squared error
    dw = np.mean(2 * error * x)      # 4. slope of loss for w
    db = np.mean(2 * error)          #    slope of loss for b
    w = w - lr * dw                  # 5. step downhill, both knobs
    b = b - lr * db
    if epoch % 100 == 0:
        print(epoch, "w =", round(w, 3), "b =", round(b, 3), "loss =", round(loss, 4))

print("Learned: y =", round(w, 2), "x +", round(b, 2))
