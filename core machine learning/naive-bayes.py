from sklearn.naive_bayes import GaussianNB

# Train
nb = GaussianNB()
nb.fit(X_scaled, y)

# Plot
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=y, edgecolor='k')
plt.title("Naive Bayes Decision Boundary")
plt.show()
