import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a, b = matrix[0]
	c, d = matrix[1]
	trace = a+d
	det = a*d-b*c
	discriminant = math.sqrt(trace**2 - 4*det)
	lambda1 = (trace + discriminant) / 2
	lambda2 = (trace - discriminant) / 2
	eigenvalues = [lambda1, lambda2]
	return eigenvalues