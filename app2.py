import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


# Page settings
st.set_page_config(page_title="Big Mart Sales Predictor", layout="centered")
st.title("🏪 Big Mart Product Sales Predictor")

# ==========================
# 📌 SIDEBAR: Model Info
# ==========================
with st.sidebar:
    st.header("📊 Model Details")
    st.markdown("""
    **🔍 Model:** XGBoost Regressor  
    **📌 Task:** Predict `Item_Outlet_Sales`  
    **📂 Inputs:**  
    - Product Features (Weight, MRP, etc.)  
    - Outlet Details (Type, Size, etc.)  

    **⚙️ Evaluation Metrics:**  
    - RMSE ≈ 1120 (example)  
    - R² Score ≈ 0.57

    **🧠 Features Used in Training:**  
    `Item_Identifier`, `Item_Weight`, `Item_Fat_Content`,  
    `Item_Visibility`, `Item_Type`, `Item_MRP`,  
    `Outlet_Identifier`, `Outlet_Establishment_Year`,  
    `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`

    **🛠 How It Works:**  
    - Takes user input  
    - Encodes categorical features  
    - Predicts using trained XGBoost model  
    - Returns estimated sales value in ₹  
    """)
    st.caption("Built with ❤️ using Streamlit and XGBoost")

# ==========================
# 🔢 User Input Form
# ==========================
st.subheader("📋 Enter Product Details")

item_identifier = st.text_input("Item Identifier", "FDA15")
item_weight = st.number_input("Item Weight", min_value=0.0, max_value=30.0, value=12.5)
item_fat_content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
item_visibility = st.slider("Item Visibility", 0.0, 0.3, 0.05)
item_type = st.selectbox("Item Type", [
    'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
    'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene',
    'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']
)
item_mrp = st.slider("Item MRP (₹)", 0.0, 300.0, 120.0)
outlet_identifier = st.selectbox("Outlet Identifier", [
    'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027',
    'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019']
)
outlet_establishment_year = st.selectbox("Outlet Establishment Year", list(range(1985, 2010)))
outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
outlet_location_type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
outlet_type = st.selectbox("Outlet Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

# ==========================
# 🔮 Prediction Logic
# ==========================
if st.button("🔮 Predict Sales"):
    try:
        model = joblib.load("xgb_model.pkl")
        feature_names = joblib.load("model_features.pkl")
    except FileNotFoundError:
        st.error("❌ Model or feature file not found.")
        st.stop()

    input_dict = {
        'Item_Identifier': [item_identifier],
        'Item_Weight': [item_weight],
        'Item_Fat_Content': [item_fat_content],
        'Item_Visibility': [item_visibility],
        'Item_Type': [item_type],
        'Item_MRP': [item_mrp],
        'Outlet_Identifier': [outlet_identifier],
        'Outlet_Establishment_Year': [outlet_establishment_year],
        'Outlet_Size': [outlet_size],
        'Outlet_Location_Type': [outlet_location_type],
        'Outlet_Type': [outlet_type]
    }

    input_df = pd.DataFrame(input_dict)

    # ✅ Encode categorical columns
    categorical_cols = [
        'Item_Identifier', 'Item_Fat_Content', 'Item_Type',
        'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])

    # ✅ Handle missing columns if any
    missing_cols = [col for col in feature_names if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[feature_names]

    # ✅ Predict
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Sales: ₹{prediction:.2f}")
