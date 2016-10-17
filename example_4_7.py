import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

# Sample scatter plots
plt.scatter([1.1, 0.3], [0.3, 1.9])
plt.show()

mean1 = [0, 0]
mean2 = [3, 3]
cov = [[1.1, 0.3], [0.3, 1.9]]

# Gaussian distributions for means mu1 and mu2
x, y = np.random.multivariate_normal(mean1, cov, 1000).T
plt.plot(x, y, '.')
plt.axis('equal')
plt.show()

x, y = np.random.multivariate_normal(mean2, cov, 1000).T
plt.plot(x, y, '.')
plt.axis('equal')
plt.show()

# Contour plot of ellipses from means mu1 and mu2
x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal(mean1, cov)
plt.contourf(x, y, rv.pdf(pos))
plt.show()

rv = multivariate_normal(mean2, cov)
plt.contourf(x, y, rv.pdf(pos))
plt.show()
