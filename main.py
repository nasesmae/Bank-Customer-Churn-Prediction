import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def train_evaluate_model(model, X, y):
    """
    Trains the model and evaluates it on the test data.
    Returns the accuracy, confusion matrix, classification report, ROC AUC, and ROC curve data.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train model
    model.fit(X_train, y_train)
    # Predict probabilities and classes
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return accuracy, conf_matrix, classification_rep, roc_auc, fpr, tpr

def plot_roc_curves(model_results):
    """
    Plots the ROC curves for the given model results.
    """
    plt.figure(figsize=(10, 8))
    for model_name, result in model_results.items():
        fpr, tpr, roc_auc = result['FPR'], result['TPR'], result['ROC AUC']
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Load data
    df = pd.read_csv("data/BankChurners_preprocessed.csv")
    X = df.drop(['Attrition_Flag'], axis=1)
    y = df['Attrition_Flag']
    # Define models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),  
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
    }
    # Train and evaluate models
    model_results = {}
    for model_name, model in models.items():
        results = train_evaluate_model(model, X, y)
        model_results[model_name] = {
            'Accuracy': results[0],
            'ROC AUC': results[3],
            'Precision': results[2]['weighted avg']['precision'],
            'Recall': results[2]['weighted avg']['recall'],
            'F1-Score': results[2]['weighted avg']['f1-score'],
            'Confusion Matrix': results[1],
            'FPR': results[4],
            'TPR': results[5]
        }
    # Display results in a DataFrame
    results_df = pd.DataFrame(model_results).T
    print(results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']])
    # Plot ROC curves
    plot_roc_curves(model_results)

if __name__ == "__main__":
    main()
