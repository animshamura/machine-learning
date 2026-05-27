from sklearn.ensemble import RandomForestClassifier

# Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Feature Importance
importances = rf.feature_importances_
plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=['Feature 1', 'Feature 2'], palette="viridis")
plt.title("Random Forest Feature Importance")
plt.show()
