# Big-Mart-Sales-Prediction

This project predicts the sales of products at different Big Mart outlets using Machine Learning. The goal is to help retailers estimate future sales based on product attributes and outlet information.

🚀 Features

Built using XGBoost Regressor for accurate predictions

Interactive Streamlit Web App for real-time sales forecasting

User-friendly form to input product & outlet details

Predicts sales value in Indian Rupees (₹)

Encodes categorical features automatically

📊 Model Details

Algorithm: XGBoost Regressor

Target Variable: Item_Outlet_Sales

Evaluation Metrics:

RMSE ≈ 1120

R² Score ≈ 0.57

Training Features:

Item_Identifier, Item_Weight, Item_Fat_Content

Item_Visibility, Item_Type, Item_MRP

Outlet_Identifier, Outlet_Establishment_Year

Outlet_Size, Outlet_Location_Type, Outlet_Type

🖥 Tech Stack

Python

Pandas, NumPy, Scikit-learn

XGBoost

Streamlit

Joblib (for model persistence)
