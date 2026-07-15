# HR Intelligence Suite

## What This Project Is

HR Intelligence Suite is a machine learning system for people-analytics decision support. It bundles three independent models behind one dashboard, each answering a different question about an employee profile:

| Model | Question it answers | Type |
|---|---|---|
| **Promotion Prediction** | Is this employee likely to be promoted? | Binary classification |
| **Salary Prediction** | What is a fair salary for this profile? | Regression |
| **Attrition Prediction** | Is this employee at risk of leaving? | Binary classification |

Each model has its own dataset, features, and preprocessing pipeline. The dashboard converts friendly inputs (dropdowns, sliders, toggles) into the exact encoded feature vector each model expects, so users never enter encoded values directly.

---

## 1. Promotion Prediction

Predicts promotion likelihood from an employee's education, recruitment channel, training history, performance, tenure, and department.

**Preprocessing:** irrelevant/identifier columns dropped, missing values imputed, education and recruitment channel manually mapped to ordinal scores, department one-hot encoded (`Analytics` as the reference category), class imbalance handled with SMOTE.

**Model:** Random Forest, chosen after comparison against Logistic Regression, KNN, Naive Bayes, and SVC.

---

## 2. Salary Prediction

Estimates a fair salary from age, education level, years of experience, seniority, gender, country, and job title.

**Preprocessing:** outlier clipping on numeric fields, categorical fields one-hot encoded, all features scaled with `StandardScaler`.

**Model:** Random Forest, chosen after comparison against Linear Regression, Decision Tree, and SVR.

---

## 3. Attrition Prediction

Flags employees at risk of leaving using workload, satisfaction, compensation, and career-history signals — including engineered features like a satisfaction composite and a job-hopping index.

**Preprocessing:** outlier clipping, feature engineering, categorical fields label-encoded, class imbalance handled with SMOTE.

**Model:** Random Forest, chosen after comparison against Logistic Regression, SVM, and KNN.

---

## Design Philosophy

The person filling out the form should never have to know an encoding scheme. Every input is translated into the correct numeric/one-hot representation, in the correct order, before being handed to the model — matching exactly what each model saw during training.
