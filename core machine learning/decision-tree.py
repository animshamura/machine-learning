from sklearn.tree import DecisionTreeClassifier, plot_tree

# Train
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_scaled, y)

# Plot
plt.figure(figsize=(12,8))
plot_tree(tree, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'])
plt.title("Decision Tree Structure")
plt.show()
