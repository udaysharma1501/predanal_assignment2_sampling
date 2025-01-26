
# Credit Card Fraud Detection using Sampling and Classification Models

This project demonstrates the use of various classification models to detect fraud in a credit card dataset. It tackles class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique) and evaluates the models on multiple samples of resampled data.

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset](#dataset)
4. [Steps](#steps)
 - [1. Install Required Libraries](#1-install-required-libraries)
 - [2. Import Libraries](#2-import-libraries)
 - [3. Load the Dataset](#3-load-the-dataset)
 - [4. Explore the Dataset](#4-explore-the-dataset)
 - [5. Handle Class Imbalance with SMOTE](#5-handle-class-imbalance-with-smote)
 - [6. Create Data Samples](#6-create-data-samples)
 - [7. Define Classification Models](#7-define-classification-models)
 - [8. Train Models and Evaluate](#8-train-models-and-evaluate)
 - [9. Analyze Results](#9-analyze-results)
5. [Output](#output)
6. [Conclusion](#conclusion)

---

## Overview
This project aims to solve the problem of class imbalance in a credit card fraud dataset and evaluate the performance of different classifiers on resampled data. The primary steps include:
- Addressing class imbalance using **SMOTE**.
- Creating multiple random and systematic samples.
- Training and testing models on these samples.
- Comparing the models' accuracy using an accuracy matrix.

---

## Prerequisites
Make sure you have the following installed:
- Python 3.8 or higher
- Required Python libraries:
  ```bash
  pip install pandas scikit-learn imbalanced-learn` 

----------

## Dataset

The dataset, `Creditcard_data.csv`, contains anonymized credit card transaction records with features and a binary target variable:

-   `Class`: 1 for fraud and 0 for non-fraud.

Upload this dataset to your working directory if you're using Google Colab.

----------

## Steps

### 1. Install Required Libraries

Install all required libraries, including `imbalanced-learn`, to handle oversampling techniques like SMOTE.

`!pip install imbalanced-learn` 

----------

### 2. Import Libraries

Import necessary libraries for data handling, sampling, modeling, and evaluation:

-   `pandas` and `numpy` for data manipulation.
-   `sklearn` for modeling and evaluation.
-   `imbalanced-learn` for oversampling.


`import pandas as pd`
`from sklearn.model_selection import train_test_split`
`from sklearn.metrics import accuracy_score`
`from sklearn.linear_model import LogisticRegression`
`from sklearn.tree import DecisionTreeClassifier`
`from sklearn.ensemble import GradientBoostingClassifier`
`from sklearn.svm import SVC`
`from sklearn.neighbors import KNeighborsClassifier`
`from imblearn.over_sampling import SMOTE`
`import numpy as np` 

----------

### 3. Load the Dataset

Upload and load the credit card dataset. Use `pandas` to read and display basic information:

`from google.colab import files`

# Upload the dataset
`uploaded = files.upload()`

# Load the CSV
`data = pd.read_csv("Creditcard_data.csv")`
`print(data.head())`
`print(data.info())` 

----------

### 4. Explore the Dataset

Inspect the dataset's structure, including class distribution, to understand the imbalance:


`print(data['Class'].value_counts())` 

----------

### 5. Handle Class Imbalance with SMOTE

Apply **SMOTE** to oversample the minority class (fraudulent transactions) and balance the dataset:


`X = data.drop("Class", axis=1)`
`y = data["Class"]`

# Apply SMOTE for oversampling
`smote = SMOTE(random_state=42)`
`X_resampled, y_resampled = smote.fit_resample(X, y)`

`print("Original class distribution:")`
`print(y.value_counts())`
`print("Resampled class distribution:")`
`print(y_resampled.value_counts())` 

----------

### 6. Create Data Samples

Generate five unique samples from the resampled data using random and systematic sampling techniques:

`sample_size = 1000`

`samples = {`
   ` "Sampling1": X_resampled.sample(n=sample_size, random_state=42),`
   `"Sampling2": X_resampled.sample(n=sample_size, random_state=21),`
    `"Sampling3": X_resampled.iloc[::len(X_resampled)//sample_size, :],`
    `"Sampling4": X_resampled.sample(n=sample_size, random_state=56),`
    `"Sampling5": X_resampled.sample(n=sample_size, random_state=99),`
`}`

`sample_datasets = {`
    `name: (sample, y_resampled.loc[sample.index])`
    `for name, sample in samples.items()`
`}` 

----------

### 7. Define Classification Models

Define five classification models for evaluation:

1.  **Logistic Regression**
2.  **Decision Tree**
3.  **Gradient Boosting Classifier**
4.  **Support Vector Classifier (SVC)**
5.  **K-Nearest Neighbors (KNN)**


`models = {`
    `"M1": LogisticRegression(random_state=42),`
    `"M2": DecisionTreeClassifier(random_state=42),`
    `"M3": GradientBoostingClassifier(random_state=42),`
    `"M4": SVC(random_state=42),`
    `"M5": KNeighborsClassifier(),`
`}` 

----------

### 8. Train Models and Evaluate

For each sample, split the data into training and testing sets and train all five models. Evaluate their accuracy on the test set using `accuracy_score` from `sklearn`:

`results = {}`

`for sample_name, (X_sample, y_sample) in sample_datasets.items():`
   ` print(f"Evaluating for {sample_name}...")`
    `X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample,`
    `test_size=0.3, random_state=42)`

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        if sample_name not in results:
            results[sample_name] = {}
        results[sample_name][model_name] = accuracy
----------

### 9. Analyze Results

Store the results in a matrix and identify the best-performing sampling technique for each model:

`matrix_data = []`
`for sample_name, accuracies in results.items():`
    `row = [accuracies.get(model, None) for model in models.keys()]`
    `matrix_data.append(row)`

`results_matrix = pd.DataFrame(`
    `matrix_data,`
    `index=results.keys(),`
    `columns=models.keys()`
`)`

`print("Accuracy Matrix:")`
`print(results_matrix)`

# Save results to a CSV file
`results_matrix.to_csv("results_matrix_colab.csv")`

`best_combinations = results_matrix.idxmax()`
`print("Best Sampling Technique for Each Model:")`
`print(best_combinations)` 

----------

## Output

**Accuracy Matrix:** A table comparing the accuracy of each model across different sampling techniques.
**Best Sampling Technique:** A summary of the best sampling strategy for each model.

----------

## Conclusion

This project demonstrates the importance of handling class imbalance in datasets and the impact of sampling strategies on model performance. By using techniques like SMOTE and comparing multiple models, we can improve the detection of fraudulent transactions effectively.

----------

## Author

Uday Sharma  
[GitHub Profile](https://github.com/udaysharma1501/predanal_assignment2_sampling)
