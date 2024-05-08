
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_evaluate_classifier(model, X, y):
    # Train the model
    model.fit(X, y)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

    # Make predictions
    y_pred = model.predict(X)

    # Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    confusion = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    return accuracy, confusion, report, cv_scores


df = pd.read_csv('data\BankChurners_preprocessed.csv')

# Define the feature matrix X and the target vector y
X = df.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)
y = df['Attrition_Flag']

# Create instances of different classifiers
logistic_regression = LogisticRegression()
knn_classifier = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()

# Train and evaluate each classifier using the function
models = [logistic_regression, knn_classifier, decision_tree, random_forest]
for model in models:
    accuracy, confusion, report, cv_scores = train_evaluate_classifier(model, X, y)
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", np.mean(cv_scores))
    print("\n")
