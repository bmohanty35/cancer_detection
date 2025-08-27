# Hospital Readmission Prediction (UCI Diabetes Dataset)
## Project Overview

This project predicts whether a patient will be readmitted to the hospital using the UCI Diabetes dataset. The dataset contains detailed information about hospital visits, patient demographics, diagnoses, and treatment details. The primary target is readmission status (yes/no).

We build a Random Forest classifier, handle data imbalance with SMOTE, optimize performance using GridSearchCV, and finally perform subgroup analysis (male vs. female patients).

## Steps in the Project
### 1. Import Libraries

We use essential Python libraries such as pandas, numpy, scikit-learn, and imbalanced-learn (SMOTE) to process the dataset and train the model.

### 2. Load Dataset

The dataset comes from the UCI Machine Learning Repository (Diabetes dataset).

Data is stored in a ZIP file and directly loaded into a Pandas DataFrame.

### 3. Preprocessing

Missing values (?) are replaced with NaN.

Columns with excessive missing data or irrelevant identifiers (like patient IDs, encounter ID, weight, payer_code, and medical_specialty) are dropped.

Target variable readmitted is simplified:

<30 or >30 → 1 (Readmitted)

NO → 0 (Not Readmitted)

Categorical variables are one-hot encoded into numeric features.

### 4. Feature & Target Definition

Features (X): All patient-related predictors.

Target (y): Readmission status (0 or 1).

### 5. Train-Test Split

The dataset is split into 70% training and 30% testing.

Stratification ensures class balance between training and testing sets.

### 6. Handle Class Imbalance with SMOTE

Since readmission cases are much fewer than non-readmission, the data is imbalanced.

SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples of the minority class to balance training data.

### 7. Model Training & Hyperparameter Tuning

Base model: Random Forest Classifier.

GridSearchCV is applied to tune hyperparameters such as:

Number of trees (n_estimators)

Tree depth (max_depth)

Minimum samples per split and leaf

The best model is chosen using F1-score as the evaluation metric.

### 8. Model Evaluation

Predictions are made on the test set.

Evaluation metrics include:

Confusion Matrix

Precision, Recall, F1-score (Classification Report)

### 9. Subgroup Analysis (Fairness Check)

To ensure fairness across genders, the model’s performance is separately evaluated on:

Male patients (where gender_Male = 1)

Female patients (where gender_Male = 0)

This helps verify if the model performs equally well for both groups.

## Key Takeaways

Balanced Training Data → SMOTE prevents bias toward the majority class.

Optimized Random Forest → GridSearchCV finds the best hyperparameters for performance.

Fairness Check → Subgroup evaluation ensures the model does not disproportionately misclassify patients based on gender.
