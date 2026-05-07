import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ──────────────────────────────────────────────
# 1. LOAD & CLEAN
# ──────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\DELL\Downloads\Projects-AIML\Projects-AIML\Datasets\project 4\vendor_dirty.csv")

numeric_cols = [
    "Total_Orders", "Delivered_Orders", "Delayed_Orders", 
    "Disputed_Orders", "Returned_Orders", "Rating",
    "Product_Quality_Score", "Response_Time_Hours", 
    "Refund_Rate", "Avg_Delivery_Days", "Years_In_Business", 
    "Contract_Value"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.median(numeric_only=True)).fillna(0)

# ──────────────────────────────────────────────
# 2. CREATE TARGET FORMULA
# ──────────────────────────────────────────────
epsilon = 0.0001
df["delay_rate"] = df["Delayed_Orders"] / (df["Total_Orders"] + epsilon)
df["dispute_rate"] = df["Disputed_Orders"] / (df["Total_Orders"] + epsilon)

# Create the hidden rule
df["risk_score"] = (
    0.35 * df["delay_rate"] +
    0.30 * df["dispute_rate"] +
    0.20 * df["Refund_Rate"] +
    0.15 * (5 - df["Rating"]) / 5
)

# Convert to 1/0 Target
df["Risk_Level"] = (df["risk_score"] > 0.30).astype(int)

# ──────────────────────────────────────────────
# 3. PREVENT DIRECT CHEATING (But keep raw data!)
# ──────────────────────────────────────────────
# We drop the EXACT rates used in the formula.
# BUT we KEEP Delayed_Orders, Total_Orders, etc., so the model has clues!
forbidden_cols = [
    "Vendor_ID", "Vendor_Name", 
    "risk_score", "delay_rate", "dispute_rate"
    # Notice we are NO LONGER dropping Delayed_Orders or Total_Orders!
]
df = df.drop(columns=[c for c in forbidden_cols if c in df.columns])

if "Vendor_Category" in df.columns:
    df = pd.get_dummies(df, columns=["Vendor_Category"], drop_first=True)

# ──────────────────────────────────────────────
# 4. TRAIN MODEL
# ──────────────────────────────────────────────
X = df.drop("Risk_Level", axis=1).astype(np.float32)
y = df["Risk_Level"].astype(np.int8)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest will now figure out how the raw counts interact!
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10, 
    class_weight="balanced", 
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# ──────────────────────────────────────────────
# 5. EVALUATE
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("🎯 FINAL MODEL RESULTS (With Historical Counts)")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred, digits=4))

# Let's see what the model used to get the high accuracy!
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 Most Important Features:")
print(importances.head(5))
import joblib

joblib.dump(model, "vendor_risk_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model saved successfully!")