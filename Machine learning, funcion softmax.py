import math
z = [3.0, 5.0, 8.0, 1.0, 9.0, 3.0]
z_exp = [math.exp(i) for i in z]
x = [round(i, 2) for i in z_exp]
print(x)
print(z_exp)
sum_z_exp = sum(x)
print(round(sum_z_exp, 2))
114.98
softmax = [round(i / sum_z_exp, 4) for i in z_exp]
print(softmax)
[0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]