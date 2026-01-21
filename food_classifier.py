import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# LOAD ALL THE MODELS AND SCALER
# -------------------------------

MODELS = {
    "Logistic Regression": joblib.load("Logistic_Regression.pkl"),
    "Decision Tree": joblib.load("Decision_Tree.pkl"),
    "Random Forest": joblib.load("Random_Forest.pkl"),
    "KNN": joblib.load("KNN.pkl"),
    "SVM": joblib.load("SVM.pkl"),
    "XGBoost": joblib.load("XGBoost.pkl"),
    "Gradient Boosting": joblib.load("Gradient_Boosting.pkl"),
}

scaler = joblib.load("scaler.pkl")

# Load label mapping

df_ref = pd.read_csv("synthetic_food_dataset_imbalanced.csv")
food_mapping = dict(enumerate(df_ref["Food_Name"].astype("category").cat.categories))

# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(page_title="Strict Diet Food Classifier", layout="wide")

st.title("🥗 Strict Diet Food Classification System")
st.markdown("---")
st.header("High-Precision Meal Selection for Diet Planning")
st.subheader("Business Objective")

st.markdown(""" 
This system predicts the **exact allowed food name** based on user-defined nutritional requirements.   
It is designed for **strict diet plans** where only one precise food choice is allowed.
""")

# -------------------------------
# USER INPUTS
# -------------------------------
st.header("Nutritional Requirements")

st.markdown("---")

Calories = st.slider("Calories", 50, 800, 300)
Protein = st.slider("Protein (g)", 0, 80, 20)
Fat = st.slider("Fat (g)", 0, 80, 15)
Carbs = st.slider("Carbohydrates (g)", 0, 120, 40)
Sugar = st.slider("Sugar (g)", 0, 50, 10)
Fiber = st.slider("Fiber (g)", 0, 30, 5)
Sodium = st.slider("Sodium (mg)", 0, 2000, 500)
Cholesterol = st.slider("Cholesterol (mg)", 0, 300, 50)
Glycemic_Index = st.slider("Glycemic Index", 0, 100, 50)
Water_Content = st.slider("Water Content (%)", 0, 100, 50)
Serving_Size = st.slider("Serving Size (g)", 50, 500, 150)

# Meal_Type = st.sidebar.selectbox("Meal Type", ["breakfast", "lunch", "dinner", "snack"])
# Preparation_Method = st.sidebar.selectbox("Preparation Method", ["raw", "fried", "baked", "boiled"])

# Is_Vegan = st.sidebar.radio("Vegan", ["Yes", "No"])
# Is_Gluten_Free = st.sidebar.radio("Gluten Free", ["Yes", "No"])

# -------------------------------
# MODEL SELECTION
# -------------------------------
model_name = st.selectbox(
    "Select Classification Model",
    list(MODELS.keys())
)

model = MODELS[model_name]

# -------------------------------
# FEATURE ENGINEERING (MATCH TRAINING)
# -------------------------------
input_dict = {
    "Calories": Calories,
    "Protein": Protein,
    "Fat": Fat,
    "Carbs": Carbs,
    "Sugar": Sugar,
    "Fiber": Fiber,
    "Sodium": Sodium,
    "Cholesterol": Cholesterol,
    "Glycemic_Index": Glycemic_Index,
    "Water_Content": Water_Content,
    "Serving_Size": Serving_Size
}

# One-hot encoding (manual for inference safety)
#for mt in ["breakfast", "lunch", "dinner", "snack"]:
#    input_dict[f"Meal_Type_{mt}"] = 1 if Meal_Type == mt else 0

#for pm in ["raw", "fried", "baked", "boiled"]:
#    input_dict[f"Preparation_Method_{pm}"] = 1 if Preparation_Method == pm else 0

X_input = pd.DataFrame([input_dict])

# Scaling based on the scaler

X_scaled = scaler.transform(X_input)


# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Allowed Food"):
    pred_code = model.predict(X_scaled)[0]
    food_name = food_mapping[pred_code]
    st.markdown("---")
    st.success(f"✅ **Predicted Allowed Food:** {food_name}")

    st.markdown("""
    ### Why this food?
    - Matches your nutritional constraints
    - Ensures strict diet compliance
    - Prevents manual selection errors
    """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Strict Diet Food Classification | ML-Driven Personalized Nutrition")
