for i in range(1, 101):
    print(f"epoch {i} loss: {1/i:.4f}")
    if i % 10 == 0:
        print(f"epoch {i} accuracy: {i/100:.2f}")
