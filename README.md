# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using historical transaction data. The project demonstrates the complete workflow of fraud detection, including data preprocessing, exploratory data analysis (EDA), handling class imbalance, model training, and performance evaluation.

---

## Project Overview

Credit card fraud is a significant challenge for financial institutions, leading to substantial financial losses each year. Machine learning models can analyze transaction patterns and identify suspicious activities in real time, helping reduce fraud and improve transaction security.

This project focuses on building a binary classification model to distinguish between legitimate and fraudulent transactions.

---

## Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Handling Class Imbalance
- Feature Scaling
- Machine Learning Model Training
- Model Evaluation
- Fraud Prediction for New Transactions

---

## Dataset

The dataset contains anonymized credit card transaction records along with their corresponding fraud labels.

Typical features include:

- Transaction Time
- Transaction Amount
- Principal Components (V1–V28)
- Transaction Class (Target Variable)
  - `0` → Legitimate Transaction
  - `1` → Fraudulent Transaction

Example:

| Time | Amount | V1 | V2 | ... | Class |
|------|--------|----|----|-----|-------|
| 0 | 149.62 | -1.36 | -0.07 | ... | 0 |
| 406 | 2.69 | -2.31 | 1.95 | ... | 1 |

> Replace this section with the actual dataset source if applicable.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Models

Depending on the notebook, the project may include:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost
- Isolation Forest (if used)

---

## Project Structure

```
.
├── Credit_Card_Fraud_Detection.ipynb
├── README.md
├── dataset/
│   └── creditcard.csv
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

Navigate to the project directory:

```bash
cd credit-card-fraud-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
Credit_Card_Fraud_Detection.ipynb
```

---

## Workflow

1. Load the dataset
2. Explore and clean the data
3. Analyze class distribution
4. Handle class imbalance
5. Scale numerical features
6. Split the dataset into training and testing sets
7. Train machine learning models
8. Evaluate model performance
9. Predict fraudulent transactions

---

## Example Prediction

**Input**

```
Transaction Amount: $250
Transaction Time: 56000
Transaction Features: V1–V28
```

**Predicted Output**

```
Fraudulent Transaction
```

---

## Results

The classification model predicts whether a transaction is legitimate or fraudulent using historical transaction patterns.

Example evaluation metrics:

| Metric | Score |
|---------|-------|
| Accuracy | 99.8% |
| Precision | 94% |
| Recall | 91% |
| F1-Score | 92% |
| ROC-AUC | 98% |

> Replace these values with the actual results from your notebook.

---

## Visualizations

The notebook may include:

- Fraud vs Legitimate Transaction Distribution
- Transaction Amount Distribution
- Correlation Heatmap
- Feature Distribution
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

---

## Future Improvements

- Hyperparameter tuning
- Advanced anomaly detection techniques
- Deep Learning models for fraud detection
- Real-time fraud detection pipeline
- Model deployment using Flask or FastAPI
- Integration with streaming transaction systems

---

## Requirements

Example packages:

```
python>=3.9
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## License

This project is licensed under the MIT License.

---

## Author

**Your Name**

GitHub: https://github.com/yourusername

---

## Acknowledgements

- Scikit-learn Documentation
- Pandas Documentation
- Matplotlib
- Seaborn
- Kaggle Credit Card Fraud Detection Dataset (if applicable)
- Python Open Source Community
