# Bank Customer Churn Prediction

This repository contains a series of Jupyter notebooks and Python scripts used to predict customer churn for a bank. The models explore different machine learning techniques to analyze customer behavior and identify potential churners.

## ⚠️ **Data Sources**

The data utilized in this project is sourced from Kaggle's [Credit Card Customers](https://www.kaggle.com/sakshigoyal7/credit-card-customers).

## 📂 Repo Structure

```
├── data
│ ├── BankChurners.csv
│ ├── BankChurners_preprocessed.csv
│ 
├── 1. preprocessing.ipynb
├── 2.1 KNN.ipynb
├── 2.2 logistic regression.ipynb
├── 2.3 decision tree.ipynb
├── 2.4 random forest.ipynb
├── 3. Clustering.ipynb
├── main.py
├── model comparison.py
├── README.md
├── requirements.txt

```


## 🔧 Installation

To run the notebooks, you will need Jupyter Notebook or JupyterLab installed. You can start the application with the following command:

```bash
jupyter notebook
```
For implementation with custom data, clone the repository and follow the instructions detailed in the `requirements.txt` file.


## 📝 Data Description

The datasets contain various features related to customer behavior and demographic information, which are used to predict the likelihood of a customer leaving the bank. The preprocessing.ipynb notebook details the steps to clean and prepare the data. 


### Preprocessing Data

The dataset named `BankChurners.csv`, provided by the client from Kaggle, underwent significant preprocessing. Key steps included:
- Dropping three columns as advised by the client: the last two columns and the client number column, which were irrelevant for the analysis.
- Handling 'Unknown' values appropriately based on the chosen modelling approach to maintaining data integrity.
- Employing label encoding to transform categorical string values into integers, facilitating effective model training.


## 🚀 Model Performance Metrics

Each model notebook (2.1 to 2.4) details specific machine learning techniques, with model comparison.ipynb providing a comprehensive comparison of their performance.

The following table summarizes the performance of various models evaluated in this project. Metrics reported include Accuracy, Precision, Recall, F1-Score, and ROC AUC.

| Model                | Accuracy | Precision | Recall  | F1-Score | ROC AUC |
|----------------------|----------|-----------|---------|----------|---------|
| KNN                  | 0.901    | 0.905     | 0.901   | 0.901    | 0.949   |
| Decision Tree        | 0.951    | 0.951     | 0.951   | 0.951    | 0.951   |
| Logistic Regression  | 0.815    | 0.815     | 0.815   | 0.815    | 0.897   |
| Random Forest        | 0.979    | 0.979     | 0.979   | 0.979    | 0.998   |

These metrics were calculated using a hold-out validation method where the dataset was split into training and test sets. The ROC AUC scores provide insight into the models' ability to distinguish between classes.
``

## Comparison models by ROC-curve
![image](assets/Comparison_models_by_ROC_AUC.png)

## Feature importance
This highlights the features that have the most significant impact on the model's performance.

![image](assets/feature importance.png)


## ⏰ Timeline

This project was completed within a timeframe of 5 working days.


