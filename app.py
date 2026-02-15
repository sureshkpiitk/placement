import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------
st.set_page_config(page_title="ML Model Deployment App", layout="wide")

st.title("📌 Indian Engineering College Placement Prediction")
st.write("Upload test dataset, select a model, and view evaluation results.")

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
        "Decision Tree": joblib.load("models/decision_tree.joblib"),
        "kNN": joblib.load("models/knn.joblib"),
        "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
        "Random Forest": joblib.load("models/random_forest.joblib"),
        "XGBoost": joblib.load("models/xgboost.joblib")
    }
    return models


models = load_models()

# ---------------------------------------------------
# Feature 1: Dataset Upload Option (CSV)
# ---------------------------------------------------
st.header("📂 Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload your test dataset CSV file", type=["csv"])

df = pd.read_csv("data/test_dataset.csv")  # default test dataset if no upload
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Uploaded Successfully!")

    st.subheader("📊 Preview of Uploaded Dataset")
    st.dataframe(df.head())

    # ---------------------------------------------------
    # Identify target column
    # ---------------------------------------------------
    target_col = "placement_status"
    

    if target_col:
        y_test = df[target_col]
        X_test = df.drop(columns=[target_col])

        # ---------------------------------------------------
        # Feature 2: Model Selection Dropdown
        # ---------------------------------------------------
        st.header("🤖 Select Machine Learning Model")
        selected_model_name = st.selectbox("Choose a model for prediction", list(models.keys()))

        model = models[selected_model_name]

        # ---------------------------------------------------
        # Prediction
        # ---------------------------------------------------
        print(X_test.columns)
        print("count", len(X_test), len(y_test))
        y_pred = model.predict(X_test)

        # ---------------------------------------------------
        # Feature 3: Display Evaluation Metrics
        # ---------------------------------------------------
        st.header("📌 Model Evaluation Metrics")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")

        # ---------------------------------------------------
        # Feature 4: Confusion Matrix / Classification Report
        # ---------------------------------------------------
        st.header("📌 Confusion Matrix & Classification Report")

        cm = confusion_matrix(y_test, y_pred)
        st.subheader("🔷 Confusion Matrix")
        st.write(cm)

        st.subheader("📄 Classification Report")
        report = classification_report(y_test, y_pred, zero_division=0)
        st.text(report)

        # ---------------------------------------------------
        # Optional: Download Predictions
        # ---------------------------------------------------
        st.header("⬇ Download Predictions")
        result_df = df.copy()
        result_df["Predicted_Label"] = y_pred

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

else:
    st.warning("⚠ Please upload a CSV file to proceed.")