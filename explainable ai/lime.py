import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

# Train a model
model = RandomForestClassifier().fit(X_train, y_train)

# Explain a prediction
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)
exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=4)

# Show explanation in notebook
exp.show_in_notebook()
