import time

for i in range(1, 31):
    print(f"epoch {i} loss: {1/i:.4f}")
    time.sleep(1)
print("DONE")
