# 🩺 Diabetes Prediction using Machine Learning

This project focuses on building and evaluating multiple machine learning models to predict the likelihood of diabetes based on diagnostic measurements. The dataset used is the **Pima Indians Diabetes Dataset**.

## 📊 Dataset

- Source: [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- Features:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- Target:
  - `Outcome`: 0 (Non-diabetic) or 1 (Diabetic)

## ⚙️ Workflow

1. **Load & Prepare Data**
   - Replaced invalid zeros with NaN for key columns.
   - Filled missing values using column-wise median.

2. **Preprocessing**
   - Standardized features using `StandardScaler`.

3. **Model Training**
   - Trained and evaluated the following models:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest
     - XGBoost

4. **Evaluation**
   - Accuracy and classification report (precision, recall, F1-score) were used for evaluation.
