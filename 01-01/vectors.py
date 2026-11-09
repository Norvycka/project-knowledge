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

print(dot([9, 1], [8, 2]))

users = [
    [0.8, 0.2],   # Ana
    [0.1, 0.9],   # Ben
]

A = [9, 1]
print(matvec(users, A))

