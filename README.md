# Commercial Mispricing Classifier — Interactive Streamlit App

This Streamlit app allows commercial and finance users to manually enter the key deal characteristics (e.g., discount, price, margin, segment, etc.) and predict whether a sales transaction is mispriced — with an associated confidence score.

## Features

- Interactive form for manual data entry
- Predicts whether the deal is "✅ Good Deal" or "🚨 Mispriced"
- Shows confidence level (e.g., 92% certain it is mispriced)
- Uses a trained Decision Tree model (`base_model_v2.pkl`)
