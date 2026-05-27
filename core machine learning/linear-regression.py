import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D

# Generate 3D regression data
X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
reg = LinearRegression().fit(X, y)

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', alpha=0.6)

# Plane
xx, yy = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 10),
                     np.linspace(X[:,1].min(), X[:,1].max(), 10))
zz = reg.intercept_ + reg.coef_[0]*xx + reg.coef_[1]*yy
ax.plot_surface(xx, yy, zz, alpha=0.4, color='blue')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
plt.title('Linear Regression Fit')
plt.show()
