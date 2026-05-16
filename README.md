# 💳 Credit Card Fraud Detection System

A real-time Machine Learning based Credit Card Fraud Detection System using Python, Scikit-learn, FastAPI, and Streamlit.

This project detects potentially fraudulent credit card transactions using imbalanced classification techniques and provides:

* Real-time fraud prediction API
* Interactive dashboard
* Fraud probability analysis
* Evaluation visualizations
* ML deployment workflow

---

# 🚀 Features

✅ Machine Learning Fraud Detection
✅ Real-Time Prediction API using FastAPI
✅ Interactive Dashboard using Streamlit
✅ SMOTE Imbalance Handling
✅ Random Forest Classifier
✅ Fraud Probability Scoring
✅ Confusion Matrix Visualization
✅ ROC Curve Visualization
✅ Precision-Recall Curve
✅ End-to-End ML Pipeline

---

# 📌 Problem Statement

Credit card fraud causes major financial losses for banks and fintech companies.

## The challenge:

* Fraud transactions are extremely rare
* Fraud patterns constantly change
* Traditional rule-based systems are not enough

This project uses Machine Learning to identify suspicious transactions based on transaction behavior and feature patterns.

---

# 🧠 Machine Learning Workflow

```text
Transaction Data
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
SMOTE Imbalance Handling
       ↓
Model Training
       ↓
Fraud Prediction
       ↓
FastAPI Backend
       ↓
Streamlit Dashboard
```

---

# 🛠️ Tech Stack

## Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* SMOTE

## Backend

* FastAPI
* Uvicorn

## Dashboard

* Streamlit

## Visualization

* Matplotlib
* Seaborn

---

# 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app/
│   └── api.py
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── fraud_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pr_curve.png
│
├── dashboard.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

Dataset used:

Kaggle Credit Card Fraud Detection Dataset

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Dataset Information

* 284,807 transactions
* Highly imbalanced dataset
* Fraud cases are very rare
* PCA-transformed features (V1–V28)

## Target Variable

| Value | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Anand1745/Credit-Card-Fraud-Detection-System.git
```

## Move Into Project Folder

```bash
cd Credit-Card-Fraud-Detection-System
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Download

Download dataset manually from:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place:

```text
creditcard.csv
```

inside:

```text
data/
```

---

# ▶️ Run Machine Learning Pipeline

```bash
python main.py
```

This will:

* preprocess dataset
* apply SMOTE
* train model
* generate evaluation graphs
* save trained model

---

# 🚀 Run FastAPI Backend

```bash
uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# 🎨 Run Dashboard

```bash
streamlit run dashboard.py
```

Dashboard opens at:

```text
http://localhost:8501
```

---

# 📈 Model Evaluation

The model is evaluated using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

## ROC-AUC Score

```text
0.97+
```

---

# 🔍 Fraud Detection Logic

The model analyzes:

* transaction patterns
* abnormal behavior
* PCA feature relationships
* transaction anomalies

Then predicts:

* legitimate transaction
* fraudulent transaction

along with fraud probability.

---

# 💡 Example Prediction Output

```json
{
  "prediction": 1,
  "fraud_probability": 0.91
}
```

---

# 🔥 Key Highlights

✅ End-to-End ML System
✅ Real-Time Prediction
✅ API Integration
✅ Interactive Dashboard
✅ Banking Analytics Use Case
✅ Deployment Workflow
✅ Resume-Ready Project

---

# 📚 Future Improvements

* SHAP Explainability
* Docker Deployment
* Kafka Streaming
* PostgreSQL Integration
* Cloud Deployment
* Authentication System
* Live Transaction Feed
* Next.js Frontend

---

# 👨‍💻 Author

Anand Ramesh Karunakaran

---

# ⭐ If You Like This Project

Give this repository a star ⭐
