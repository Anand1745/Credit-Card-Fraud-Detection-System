import streamlit as st
import requests
from PIL import Image

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    "Real-Time Fraud Prediction using Machine Learning + FastAPI"
)

st.divider()

# -----------------------------------
# SIDEBAR INPUT
# -----------------------------------
st.sidebar.header("Transaction Input")

features = []

# V1 to V28
for i in range(1, 29):

    value = st.sidebar.number_input(
        f"V{i}",
        value=0.0,
        step=0.1
    )

    features.append(value)

# Amount
amount = st.sidebar.number_input(
    "Amount",
    value=100.0,
    step=10.0
)

# Time
transaction_time = st.sidebar.number_input(
    "Time",
    value=5000.0,
    step=100.0
)

features.append(amount)
features.append(transaction_time)

# Predict button
predict_button = st.sidebar.button(
    "🚀 Predict Fraud"
)

# -----------------------------------
# MAIN COLUMNS
# -----------------------------------
col1, col2 = st.columns(2)

# -----------------------------------
# PREDICTION SECTION
# -----------------------------------
with col1:

    st.subheader("Prediction Output")

    if predict_button:

        try:

            url = "http://127.0.0.1:8000/predict"

            payload = {
                "data": features
            }

            response = requests.post(
                url,
                json=payload
            )

            result = response.json()

            prediction = result["prediction"]

            probability = result[
                "fraud_probability"
            ]

            # Prediction display
            if prediction == 1:

                st.error(
                    "🚨 Fraudulent Transaction Detected"
                )

            else:

                st.success(
                    "✅ Legitimate Transaction"
                )

            # Probability
            st.metric(
                label="Fraud Probability",
                value=f"{probability:.2%}"
            )

            # Progress bar
            st.progress(float(probability))

            # Risk levels
            if probability < 0.3:

                st.info("Low Risk Transaction")

            elif probability < 0.7:

                st.warning("Medium Risk Transaction")

            else:

                st.error("High Risk Transaction")

        except Exception as e:

            st.error("API Connection Failed")

            st.exception(e)

# -----------------------------------
# SYSTEM INFO
# -----------------------------------
with col2:

    st.subheader("System Information")

    st.write("### Features")

    st.write("✅ Machine Learning Model")
    st.write("✅ Fraud Detection")
    st.write("✅ Random Forest")
    st.write("✅ SMOTE")
    st.write("✅ FastAPI Backend")
    st.write("✅ Real-Time Prediction")

    st.write("### Purpose")

    st.write(
        "This system detects potentially fraudulent "
        "credit card transactions using machine learning."
    )

# -----------------------------------
# VISUALIZATION SECTION
# -----------------------------------
st.divider()

st.subheader("Model Evaluation")

viz1, viz2, viz3 = st.columns(3)

# Confusion Matrix
with viz1:

    st.write("### Confusion Matrix")

    try:

        cm_image = Image.open(
            "outputs/confusion_matrix.png"
        )

        st.image(cm_image)

    except:

        st.warning(
            "confusion_matrix.png not found"
        )

# ROC Curve
with viz2:

    st.write("### ROC Curve")

    try:

        roc_image = Image.open(
            "outputs/roc_curve.png"
        )

        st.image(roc_image)

    except:

        st.warning(
            "roc_curve.png not found"
        )

# PR Curve
with viz3:

    st.write("### Precision Recall Curve")

    try:

        pr_image = Image.open(
            "outputs/pr_curve.png"
        )

        st.image(pr_image)

    except:

        st.warning(
            "pr_curve.png not found"
        )

# -----------------------------------
# FOOTER
# -----------------------------------
st.divider()

st.markdown(
    """
    ## 📌 Tech Stack

    - Python
    - Scikit-learn
    - FastAPI
    - Streamlit
    - Random Forest
    - SMOTE
    - Fraud Analytics
    """
)