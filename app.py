import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Vendor Risk Dashboard",
    layout="wide"
)

# =====================================================
# LOAD MODEL + DATASET
# =====================================================
model = joblib.load("vendor_risk_model.pkl")
model_columns = joblib.load("model_columns.pkl")

df = pd.read_csv("vendor_dirty.csv")

# =====================================================
# CLEAN DATA
# =====================================================
numeric_cols = [
    "Total_Orders",
    "Delivered_Orders",
    "Delayed_Orders",
    "Disputed_Orders",
    "Returned_Orders",
    "Rating",
    "Product_Quality_Score",
    "Response_Time_Hours",
    "Refund_Rate",
    "Avg_Delivery_Days",
    "Years_In_Business",
    "Contract_Value"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.fillna(df.median(numeric_only=True), inplace=True)
# Fill numeric columns only
numeric_df = df.select_dtypes(include=[np.number]).columns

df[numeric_df] = df[numeric_df].fillna(0)

# Fill object/string columns separately
object_df = df.select_dtypes(include=["object", "string"]).columns

df[object_df] = df[object_df].fillna("Unknown")

# =====================================================
# TITLE
# =====================================================
st.title("AI-Based Vendor Risk Prediction Dashboard")

st.write("""
This system analyzes historical vendor performance and predicts whether a vendor is HIGH RISK or LOW RISK using Machine Learning.
""")

# =====================================================
# VENDOR SELECTION
# =====================================================
vendor_names = (
    df["Vendor_Name"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

vendor_names = sorted(vendor_names)

selected_vendor = st.selectbox(
    "Select Vendor",
    sorted(vendor_names)
)

# =====================================================
# GET SELECTED VENDOR DATA
# =====================================================
vendor_data = df[df["Vendor_Name"] == selected_vendor].copy()

if len(vendor_data) > 0:

    vendor_row = vendor_data.iloc[0]

    st.subheader("Vendor Details")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Orders", int(vendor_row["Total_Orders"]))
    col1.metric("Delayed Orders", int(vendor_row["Delayed_Orders"]))
    col1.metric("Disputed Orders", int(vendor_row["Disputed_Orders"]))

    col2.metric("Returned Orders", int(vendor_row["Returned_Orders"]))
    col2.metric("Rating", round(vendor_row["Rating"], 2))
    col2.metric("Refund Rate", round(vendor_row["Refund_Rate"], 2))

    col3.metric("Response Time", round(vendor_row["Response_Time_Hours"], 2))
    col3.metric("Delivery Days", round(vendor_row["Avg_Delivery_Days"], 2))
    col3.metric("Contract Value", round(vendor_row["Contract_Value"], 2))

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    epsilon = 0.0001

    vendor_data["delay_rate"] = (
        vendor_data["Delayed_Orders"] /
        (vendor_data["Total_Orders"] + epsilon)
    )

    vendor_data["dispute_rate"] = (
        vendor_data["Disputed_Orders"] /
        (vendor_data["Total_Orders"] + epsilon)
    )

    vendor_data["return_rate"] = (
        vendor_data["Returned_Orders"] /
        (vendor_data["Total_Orders"] + epsilon)
    )

    # =====================================================
    # ENCODE CATEGORY
    # =====================================================
    if "Vendor_Category" in vendor_data.columns:

        vendor_data["Vendor_Category"] = (
            vendor_data["Vendor_Category"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        vendor_data = pd.get_dummies(
            vendor_data,
            columns=["Vendor_Category"],
            drop_first=True
        )

    # =====================================================
    # REMOVE UNUSED COLUMNS
    # =====================================================
    drop_cols = [
        "Vendor_ID",
        "Vendor_Name",
        "Risk_Level",
        "risk_score"
    ]

    vendor_data.drop(
        columns=[c for c in drop_cols if c in vendor_data.columns],
        inplace=True,
        errors="ignore"
    )

    # =====================================================
    # MATCH TRAINING COLUMNS
    # =====================================================
    for col in model_columns:
        if col not in vendor_data.columns:
            vendor_data[col] = 0

    vendor_data = vendor_data[model_columns]

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================
    if st.button("Predict Vendor Risk"):

        prediction = model.predict(vendor_data)[0]

        probability = model.predict_proba(vendor_data)[0]

        confidence = max(probability) * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                f"⚠ HIGH RISK Vendor ({confidence:.2f}% confidence)"
            )
        else:
            st.success(
                f"✅ LOW RISK Vendor ({confidence:.2f}% confidence)"
            )

        # =====================================================
        # PROBABILITY BREAKDOWN
        # =====================================================
        st.subheader("Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Risk Level": ["Low Risk", "High Risk"],
            "Probability": probability
        })

        st.bar_chart(
            prob_df.set_index("Risk Level")
        )

        # =====================================================
        # FEATURE IMPORTANCE
        # =====================================================
        st.subheader("Top Important Features")

        importance = pd.Series(
            model.feature_importances_,
            index=model_columns
        ).sort_values(ascending=False).head(10)

        st.bar_chart(importance)
