# DA5401 – Assignment 7  
### Multi-Class Model Selection using ROC and Precision–Recall Curves  
**Name:** Omkar Ashok Chaudhari  
**Roll No:** NA22B059  

---

## 1. Objective

The goal of this assignment is to perform **model selection** for a multi-class classification problem using the **UCI Landsat Satellite (Statlog)** dataset.  
The focus is on analyzing model performance not just through standard accuracy metrics but through **Receiver Operating Characteristic (ROC)** and **Precision–Recall Curves (PRC)** — two threshold-dependent evaluation tools that reveal deeper insights into classifier behavior.

---

## 2. Dataset Description

- **Dataset:** UCI Landsat Satellite (Statlog) Dataset  
- **Classes:** 6 primary land-cover types (with labels {1, 2, 3, 4, 5, 7})  
- **Features:** 36 continuous numerical attributes per sample  
- **Total Samples:** 6,435 (combined from `sat.trn` and `sat.tst`)  

The dataset was loaded, combined, and standardized using **StandardScaler** before model training.

---

## 3. Models Implemented

Six baseline models (as per the assignment instructions):

| No. | Model | sklearn Class | Key Details |
|-----|--------|----------------|--------------|
| 1 | K-Nearest Neighbors (KNN) | `KNeighborsClassifier()` | Non-parametric, distance-based |
| 2 | Decision Tree | `DecisionTreeClassifier()` | Interpretable but prone to overfitting |
| 3 | Dummy Classifier (Prior) | `DummyClassifier(strategy="prior")` | Random baseline (AUC ≈ 0.5) |
| 4 | Logistic Regression | `LogisticRegression(max_iter=1000)` | Linear, well-calibrated |
| 5 | Gaussian Naive Bayes | `GaussianNB()` | Simple probabilistic baseline |
| 6 | Support Vector Classifier (SVC) | `SVC(probability=True)` | Non-linear kernel-based model |

Two ensemble models were added for the optional *Brownie Points* section:
- **Random Forest** (`RandomForestClassifier`)  
- **XGBoost** (`XGBClassifier`, with label remapping for contiguous class indexing)

Additionally, a **deliberately bad model (inverted SVC)** was used to demonstrate a classifier with **AUC < 0.5**.

---

## 4. Methodology and Implementation Steps

### Step 1: Data Preparation
- Loaded `sat.trn` and `sat.tst` from the UCI repository.  
- Combined and standardized the dataset.  
- Split into **80% training** and **20% testing** sets with **stratified sampling**.

### Step 2: Model Training
- Trained all six baseline models using default hyperparameters.  
- Stored trained models in a dictionary for consistent evaluation.

### Step 3: Baseline Evaluation
- Computed **Accuracy** and **Weighted F1-score** for all models.  
- **KNN (0.91 F1)** and **SVC (0.89 F1)** emerged as top performers.  
- **Dummy Classifier** provided the random baseline (F1 = 0.09).

### Step 4: ROC Analysis
- Used **One-vs-Rest (OvR)** strategy for multi-class ROC computation.  
- Computed **macro-averaged AUC** for all models.  
- **KNN** and **SVC** had the highest AUC (~0.98).  
- **Dummy Classifier** had AUC = 0.5 (random performance).  
- ROC curves clearly separated good and poor models.

### Step 5: PRC Analysis
- Computed **macro-averaged Precision–Recall Curves** and **Average Precision (AP)**.  
- Observed precision-recall trade-offs at different thresholds.  
- **KNN (AP = 0.92)** and **SVC (AP = 0.90)** again outperformed others.  
- Poor models (Decision Tree, Naive Bayes) showed sharp precision drops at high recall.

### Step 6: Final Comparison and Recommendation
- Compared rankings across F1, ROC-AUC, and PRC-AP.  
- All metrics aligned closely, confirming:
  - **Best Model:** KNN  
  - **Runner-Up:** SVC  
  - **Strong Linear Baseline:** Logistic Regression  
- KNN was chosen for its superior balance between discrimination ability, calibration, and simplicity.

### Step 7: Brownie Points (Optional)
- **Random Forest** and **XGBoost** were trained and evaluated.  
  - **XGBoost** achieved the best overall performance (AUC = 0.989, AP = 0.947).  
  - **Random Forest** performed comparably (AUC = 0.987, AP = 0.939).  
- **Inverted SVC** confirmed the concept of AUC < 0.5, producing AUC = 0.02.

---

## 5. Key Insights

- ROC and PRC analyses reveal **ranking quality and threshold sensitivity**, which accuracy alone cannot.  
- KNN and SVC consistently rank highest across all metrics, demonstrating strong multi-class discrimination.  
- Ensemble models (Random Forest, XGBoost) provide superior calibration and precision stability.  
- The Dummy and inverted models illustrate the importance of comparing against a **random baseline**.

---

## 6. Final Model Recommendation

| Category | Recommended Model | Justification |
|-----------|------------------|----------------|
| Baseline | **K-Nearest Neighbors (KNN)** | Highest F1, AUC, and AP across core metrics |
| Ensemble (Optional) | **XGBoost** | Best overall precision–recall balance and probability calibration |
| Theoretical Case | **Inverted SVC** | Demonstrates concept of AUC < 0.5 (worse than random) |

---

## 7. Deliverables Included

1. Jupyter Notebook with:
   - Well-documented **markdown explanations**
   - Clean, reproducible **Python code cells**
   - **ROC** and **PRC** visualizations
   - Baseline and ensemble model comparisons  
2. This README file summarizing the entire workflow and findings.

---

## 8. Conclusion

This assignment demonstrates the complete workflow for **multi-class model evaluation using ROC and PRC curves**.  
By combining standard metrics (F1, Accuracy) with threshold-dependent measures (AUC, AP), the analysis provides a comprehensive view of classifier reliability.  
KNN proved to be the most balanced baseline model, while XGBoost surpassed all others under ensemble methods, validating the value of boosting in complex classification tasks.
