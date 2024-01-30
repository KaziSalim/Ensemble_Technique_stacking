
# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('lr', LogisticRegression(random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

# Define the meta-model
meta_model = LogisticRegression()

# Create the stacking classifier
stacking_classifier = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train the stacking classifier
stacking_classifier.fit(X_train, y_train)

# Make predictions on the test set
stacking_predictions_test = stacking_classifier.predict(X_test)

# Make predictions on the train set
stacking_predictions_train = stacking_classifier.predict(X_train)

# Evaluate the performance train
accuracy = accuracy_score(y_train, stacking_predictions_train)
print(f"Stacking Classifier Accuracy: {accuracy:.2f}")

# Evaluate the performance test
accuracy = accuracy_score(y_test, stacking_predictions_test)
print(f"Stacking Classifier Accuracy: {accuracy:.2f}")

# Cross-validation to assess generalization performance
cv_accuracy = cross_val_score(stacking_classifier, X, y, cv=5, scoring='accuracy').mean()
print(f"Cross-validated Accuracy: {cv_accuracy:.2f}")
