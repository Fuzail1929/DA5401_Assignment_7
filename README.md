- Name : Mohd Fuzail 
- Roll no. CH22B080
- Date :24 -12 -2025

# 🛰️ DA5401 A7: Multi-Class Model Selection using ROC and PRC

## 📘 Introduction
This project focuses on **multi-class model selection** using the **Landsat Satellite dataset**.  
The task is to classify different **land cover types** and evaluate various machine learning classifiers using **Receiver Operating Characteristic (ROC)** and **Precision–Recall Curve (PRC)** analysis.  
The goal is to identify the best-performing model based on its overall **accuracy**, **F1-score**, **AUC**, and **Average Precision (AP)**.

---

## 🎯 Objectives
- Load and preprocess the Landsat dataset.  
- Train multiple classifiers on the same dataset.  
- Evaluate models using:
  - Accuracy and Weighted F1-Score  
  - ROC-AUC (One-vs-Rest macro averaging)  
  - Precision–Recall Curve (macro averaging)  
- Interpret and compare performance across metrics.  
- Recommend the best model with balanced precision and recall.  

---

## 🧠 Models Used
| Model | Library | Expected Performance |
|--------|----------|----------------------|
| K-Nearest Neighbors (KNN) | `sklearn.neighbors` | Moderate / Good |
| Decision Tree | `sklearn.tree` | Moderate |
| Dummy (Prior) | `sklearn.dummy` | Baseline (AUC < 0.5 expected) |
| Logistic Regression | `sklearn.linear_model` | Good / Baseline |
| Gaussian Naive Bayes | `sklearn.naive_bayes` | Poor / Varies |
| Support Vector Classifier (SVC) | `sklearn.svm` | Strong performer |
| Random Forest (Bonus) | `sklearn.ensemble` | High |
| XGBoost (Bonus) | `xgboost` | Very High |

---

## ⚙️ Implementation Steps
1. **Data Preparation**
   - Load Landsat dataset from UCI Repository.
   - Standardize features and split data into train/test sets.

2. **Model Training**
   - Train all six core models (and three bonus models for extra credit).

3. **Evaluation**
   - Compute Accuracy and Weighted F1-score.
   - Generate macro-averaged ROC and PRC curves using One-vs-Rest (OvR) strategy.
   - Calculate AUC and Average Precision (AP) for each model.

4. **Analysis**
   - Compare model rankings across F1, AUC, and AP metrics.
   - Interpret differences and identify trade-offs.

5. **Recommendation**
   - Select the best model based on balance between precision, recall, and robustness.

---

## 📊 Observations
- **SVC** achieved the highest **Macro ROC-AUC (≈ 0.981)** and **Average Precision (≈ 0.908)**, indicating excellent discrimination and calibration.  
- **Dummy (Prior)** performed the worst, confirming its inability to learn feature relationships.  
- **ROC curves** show ranking ability across thresholds, while **PRC curves** reveal precision–recall trade-offs, especially for less frequent classes.

---

## 🏁 Final Recommendation
> The **Support Vector Classifier (SVC)** is recommended as the best model for this classification task.  
> It delivers the most consistent and balanced performance across all evaluation metrics and thresholds.

---

## 🧩 Dependencies
- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost (for bonus task)

Install all dependencies using:
```bash
pip install numpy pandas matplotlib scikit-learn xgboost
