import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("base_model_v2.pkl")

model = load_model()

# App title and instructions
st.title("💰 Commercial Mispricing Predictor")
st.markdown("Use the form below to evaluate a deal. The model will predict whether it's **mispriced** and how confident it is.")

# Input form
with st.form("deal_form"):
    discount_pct = st.slider("Discount %", 0.0, 1.0, 0.1)
    sales_price = st.number_input("Sales Price ($)", min_value=0.0, value=500.0)
    cost_pp = st.number_input("Cost per Unit ($)", min_value=0.0, value=400.0)
    quantity = st.number_input("Quantity", min_value=1, value=1)

    category = st.selectbox("Product Category", ["Furniture", "Technology", "Office Supplies"])
    sub_category = st.selectbox("Sub-Category", ["Binders", "Chairs", "Phones", "Bookcases", "Machines"])
    region = st.selectbox("Region", ["East", "West", "Central", "South"])
    segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
    order_date_weekday = st.selectbox("Order Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    ship_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])

    profit_margin_pct = st.slider("Profit Margin %", -1.0, 1.0, 0.1)

    submit = st.form_submit_button("Predict")

# Prediction logic
if submit:
    input_df = pd.DataFrame([{
        "discount_pct": discount_pct,
        "sales_price": sales_price,
        "cost_pp": cost_pp,
        "quantity": quantity,
        "category": category,
        "sub-category": sub_category,
        "region": region,
        "segment": segment,
        "order_date_weekday": order_date_weekday,
        "ship_mode": ship_mode,
        "profit_margin_pct": profit_margin_pct
    }])

    prob = model.predict_proba(input_df)[0]
    pred_class = model.predict(input_df)[0]
    confidence = round(prob[pred_class] * 100, 1)
    label = "🚨 Mispriced" if pred_class == 1 else "✅ Good Deal"

    # Display result
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence Level:** {confidence}%")
