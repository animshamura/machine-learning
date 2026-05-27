from sklearn.neighbors import KNeighborsClassifier

# Train
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Decision boundary
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, cmap='YlGnBu', alpha=0.4)
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=y, edgecolor='k')
plt.title("KNN Decision Boundary (k=5)")
plt.show()
