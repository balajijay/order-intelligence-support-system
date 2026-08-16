from flask import Flask, request, jsonify
import mysql.connector
import pandas as pd

app = Flask(__name__)

def get_training_data():
    # Connects to your local MySQL order_db database
    db = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="bala12345",  # <-- Put your actual MySQL root password here
        database="order_db"
    )
    query = "SELECT amount, is_fraud FROM transactions"
    df = pd.read_sql(query, db)
    db.close()
    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    amount = float(data.get('amount', 0))
    amount_to_avg_ratio = float(data.get('amount_to_avg_ratio', 1.0))
    
    print(f"[Python ML] Evaluating live transaction: Ratio={amount_to_avg_ratio:.2f}, Amount=${amount}")
    
    # Simple threshold rules simulating an XGBoost branch split point
    is_fraud = 0
    if amount_to_avg_ratio > 5.0 or amount > 500:
        is_fraud = 1
        
    return jsonify({
        "fraud_probability": 0.94 if is_fraud else 0.03,
        "is_fraud": bool(is_fraud)
    })

if __name__ == '__main__':
    print("[Python ML] Simulating training sequence from MySQL data cache...")
    try:
        df = get_training_data()
        print(f"[Python ML] Engine initialized with {len(df)} historical transaction records.")
    except Exception as e:
        print(f"[Python ML] Warning (Using default thresholds): {e}")
        
    app.run(port=5000)
