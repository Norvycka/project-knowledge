import numpy as np
import matplotlib.pyplot as plt

# corners of a 1x1 square (the last corner repeats the first, to close the shape)
square = np.array([[0, 0], [1, 0], [1, 0.3], [0.3, 0.3], [0.3, 2], [0, 2], [0, 0]])


def show(M, title):
    moved = square @ M.T          # apply M to every corner at once
    plt.plot(square[:, 0], square[:, 1], "b-", label="before")
    plt.plot(moved[:, 0], moved[:, 1], "r-", label="after")
    plt.axhline(0, color="gray")
    plt.axvline(0, color="gray")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()

S = np.array([[2, 0],
              [0, 3]])
show(S, "scaling")

F = np.array([[-1, 0],
              [ 0, 1]])
show(F, "mirror")

R = np.array([[0, -1],
              [1,  0]])
show(R, "rotate 90")

S2 = np.array([[2, 0],
               [0, 0.5]])      # stretch x ×2, squash y ×0.5

show(S2 @ R, "rotate first, then stretch")
show(R @ S2, "stretch first, then rotate")
