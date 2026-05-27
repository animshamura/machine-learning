import xgboost as xgb

# Train
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_scaled, y)

# Plot Feature Importance
xgb.plot_importance(model)
plt.title("XGBoost Feature Importance")
plt.show()
