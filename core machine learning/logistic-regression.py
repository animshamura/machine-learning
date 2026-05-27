from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 2D classification data
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=3)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
clf = LogisticRegression()
clf.fit(X_scaled, y)

# Plotting
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 200),
                     np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=y, palette='RdBu', edgecolor='k')
plt.title("Logistic Regression Decision Boundary")
plt.show()
