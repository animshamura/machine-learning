import shap
import xgboost
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load data
X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X, columns=load_boston().feature_names)

# Train a model
model = xgboost.XGBRegressor().fit(X, y)

# Explain the model predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Plot the first prediction’s explanation
shap.plots.waterfall(shap_values[0])

# Or summary of feature importances
shap.plots.beeswarm(shap_values)
