import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import joblib

# Load Dataset
df = pd.read_csv("data/creditcard.csv")

print(df.head())

# Check class distribution
print(df['Class'].value_counts())

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount and Time
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("After SMOTE:")
print(y_train_smote.value_counts())

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred = model.predict(X_test)

# Probability scores
y_prob = model.predict_proba(X_test)[:,1]

# Evaluation
print(classification_report(y_test, y_pred))

print("ROC AUC Score:")
print(roc_auc_score(y_test, y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.savefig("outputs/roc_curve.png")
plt.show()

# Precision Recall Curve
precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")

plt.savefig("outputs/pr_curve.png")
plt.show()

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("Model saved successfully.")