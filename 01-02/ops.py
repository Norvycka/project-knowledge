import numpy as np

rengar = np.array([9.9, 4.0, 0.0, 0.0, 0.0])
me = np.array([9.9, 4.0, 0.0, 0.0, 0.0])

print(rengar * me)
print(rengar @ me)

champions = np.array([
    [9.9, 4.0, 0.0, 0.0, 0.0],   # rengar
    [9.9, 3.0, 0.0, 0.0, 0.0],   # talon
    [0.0, 0.0, 6.0, 0.0, 9.0],   # soraka
])
bias = np.array([-10.0, 5.0, 20.0])   # rengar nerfed, talon buffed, soraka very strong

scores = champions @ me + bias
print(scores)

two_players = np.array([
    [114.01, 110.01, 0.0],    # you
    [20.0,   30.0,  90.0],    # a Soraka main friend
])
print(two_players + bias)

M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[1, 2], [2, 4]])

print(np.linalg.det(M1))
print(np.linalg.inv(M1))
print(np.linalg.inv(M2))
