import pickle
import pandas as pd

# 1. Load the model bundle generated in Part 1
artifact_path = 'artifacts/return_risk_model.pickle'
with open(artifact_path, 'rb') as f:
    bundle = pickle.load(f)

# Extract components from the saved bundle
model_pipeline = bundle['model']
feature_columns = bundle['feature_columns']
business_threshold = bundle['threshold']

print(f"✅ Loaded model bundle successfully.")
print(f"📈 Production Business Threshold: {business_threshold:.3f}\n")

# 2. Simulate a brand-new incoming customer order
# Ensure all columns required by your Part 1 features exist here
new_order = pd.DataFrame([{
    'price': 120.00,
    'discount_pct': 15.0,
    'prev_returns': 3,
    'prev_orders': 5,
    'delivery_days': 3,
    'delivery_delayed': 1,
    'payment_method': 'COD',
    'product_category': 'Electronics',
    'customer_rating': None,  # Simulating missing data (MAR) to test pipeline robustly
    'customer_rating_missing': 0
}])

# 3. Predict the probability of return
# The pipeline automatically handles the scaling, encoding, and missing data imputation
predicted_probability = model_pipeline.predict_proba(new_order[feature_columns])[:, 1][0]

# 4. Apply the business optimized decision rule
risk_status = "⚠️ HIGH RISK" if predicted_probability >= business_threshold else "✅ LOW RISK"

# 5. Output results
print("=== NEW ORDER EVALUATION ===")
print(f"Calculated Return Probability : {predicted_probability:.1%}")
print(f"Fraud / Risk Assessment       : {risk_status}")
