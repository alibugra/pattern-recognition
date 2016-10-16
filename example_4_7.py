import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot as plt

# Sample scatter plots
plt.scatter([1.1, 0.3], [0.3, 1.9])
plt.show()

mean1 = [0, 0]
mean2 = [3, 3]
cov = [[1.1, 0.3], [0.3, 1.9]]

# Gaussian distributions for means mu1 and mu2
x, y = np.random.multivariate_normal(mean1, cov).T
plt.plot(x, y, '.')
plt.axis('equal')
plt.show()

x, y = np.random.multivariate_normal(mean2, cov).T
plt.plot(x, y, '.')
plt.axis('equal')
plt.show()
