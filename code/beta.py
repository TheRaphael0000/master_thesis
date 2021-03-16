from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

plt.figure()


a, b = 0.85, 6.0

A = 0.001
B = 0.999

x = np.linspace(A, B, 1000)
y = beta.pdf(x, a, b) - beta.pdf(x, b, a)
# normalize y
y = y / (max(y) - min(y)) * 2

print(x)
print(y)

plt.plot(x, y)
plt.savefig("beta.png")
