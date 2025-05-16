import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("base_model_v2.pkl")

model = load_model()

st.title("💰 Commercial Mispricing Predictor")
st.markdown("Use the form to input deal details. The model will predict whether it's **mispriced** — with a confidence rating.")

# Form inputs
with st.form("deal_form"):
    discount_pct = st.slider("Discount %", 0.0, 1.0, 0.1)
    sales_price = st.number_input("Sales Price ($)", min_value=0.01, value=500.0)
    cost_pp = st.number_input("Cost per Unit ($)", min_value=0.0, value=400.0)
    quantity = st.number_input("Quantity", min_value=1, value=1)

    category = st.selectbox("Category", ["Furniture", "Technology", "Office Supplies"])
    sub_category = st.selectbox("Sub-Category", ["Binders", "Chairs", "Phones", "Bookcases", "Machines"])
    region = st.selectbox("Region", ["East", "West", "Central", "South"])
    segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
    order_date_weekday = st.selectbox("Order Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    ship_mode = st.selectbox("Ship Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])

    submitted = st.form_submit_button("Predict")

# On submit
if submitted:
    profit_margin_pct = (sales_price - cost_pp) / sales_price if sales_price > 0 else 0.0

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

    proba = model.predict_proba(input_df)[0]
    pred_class = model.predict(input_df)[0]
    confidence = round(proba[pred_class] * 100, 1)
    label = "🚨 Mispriced" if pred_class == 1 else "✅ Good Deal"

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence Level:** {confidence}%")
