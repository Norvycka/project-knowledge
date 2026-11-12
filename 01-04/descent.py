w = 2.0
lr = 0.1

for step in range(20):
    slope = 2 * w
    w = w - slope * lr                 # the gradient descent rule
    print(step + 1, round(w, 4), "error:", round(w ** 2, 4))
