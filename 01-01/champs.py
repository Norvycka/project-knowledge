import numpy as np

def dot(a, b):
        total=0
        for x, y in zip(a, b):
                total= total + x*y
        return total

def matvec(M, v):
        results = []
        for row in M:
                results.append(dot(row, v))
        return results

rengar = [9.9, 4.0, 0.0, 0.0, 0.0]
talon = [9.9, 3.0, 0.0, 0.0, 0.0]
soraka = [0.0, 0.0, 6.0, 0.0, 9.0]

champions = [rengar, talon, soraka]

me = [9.9, 4.0, 0.0, 0.0, 0.0]

C = np.array(champions)
m = np.array(me)

print(dot(rengar, talon))
print(dot(rengar, soraka))

print(matvec(champions, me))

print(np.dot(rengar, talon))
print(C @ m)
