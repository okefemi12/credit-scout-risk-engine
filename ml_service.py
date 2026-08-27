import os
import numpy as np
import tensorflow as tf
import shap
import pickle
from groq import Groq

# Initialize global resources
model, scaler, columns, explainer = None, None, None, None
groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

BUSINESS_MAP = {
    'step': 'Transaction Hour', 'type_enc': 'Txn Type (Transfer/CashOut)',
    'amount': 'Transaction Amount', 'oldbalanceOrg': 'Origin Acct Balance (Pre)',
    'newbalanceOrig': 'Origin Acct Balance (Post)', 'oldbalanceDest': 'Recipient Acct Balance (Pre)',
    'newbalanceDest': 'Recipient Acct Balance (Post)', 'errorBalanceOrig': 'Origin Math Discrepancy',
    'errorBalanceDest': 'Recipient Math Discrepancy'
}

# Custom patched layer to strip quantization config during deserialization
class PatchedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        if 'quantization_config' in kwargs:
            kwargs.pop('quantization_config')
        super().__init__(*args, **kwargs)


def load_ml_resources():
    global model, scaler, columns, explainer
    print("CURRENT WORKING DIRECTORY:", os.getcwd())
    print("FILES IN DIRECTORY:", os.listdir("."))
    
    # Load model with custom objects to fix the quantization_config crash
    model = tf.keras.models.load_model(
        'latest_checkpoint.h5', 
        custom_objects={'Dense': PatchedDense},
        compile=False
    )
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f: columns = pickle.load(f)
    with open('shap_metadata.pkl', 'rb') as f: shap_data = pickle.load(f)
    explainer = shap.GradientExplainer(model, shap_data['background_sample'])


def process_transaction(req):
    type_val = 1 if req.txn_type == 'CASH_OUT' else 0
    error_bal_orig = req.old_balance_org + req.amount - req.new_balance_orig
    
    # Build feature dictionary matching the Hugging Face app structure
    feature_dict = {
        'step': 150,
        'type_enc': type_val,
        'amount': req.amount,
        'oldbalanceOrg': req.old_balance_org,
        'newbalanceOrig': req.new_balance_orig,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'errorBalanceOrig': error_bal_orig,
        'errorBalanceDest': 0.0
    }
    
    # Build array dynamically in exact column order from columns.pkl
    raw_features = np.array([feature_dict[col] for col in columns]).reshape(1, -1)
    
    # Scale features with fallback logic matching your Hugging Face implementation
    try:
        scaled = scaler.transform(raw_features)
        if np.abs(scaled).max() > 100:
            raise ValueError("Scaled values out of bounds")
    except Exception:
        manual_means = np.array([243.39, 0.5, 180000, 834000, 855000, 1100000, 1225000, 0, 0])
        manual_stds = np.array([142.3, 0.5, 604000, 2900000, 2940000, 3400000, 3670000, 380000, 420000])
        scaled = (raw_features - manual_means) / (manual_stds + 1e-8)
        
    lstm_input = scaled.reshape(1, 1, len(columns))
    
    risk_prob = float(model.predict(lstm_input, verbose=0)[0][0])
    shap_vals = explainer.shap_values(lstm_input)
    
    report = generate_llm_report(shap_vals, raw_features)
    return risk_prob, report, error_bal_orig


def generate_llm_report(shap_values, raw_features):
    if not client: return "Error: Groq API Key missing."
    
    vals = shap_values[0].flatten() if isinstance(shap_values, list) else shap_values.flatten()
    real_values = raw_features.flatten()
    
    feature_data = sorted(
        [(BUSINESS_MAP.get(columns[i], columns[i]), real_values[i], vals[i]) for i in range(len(columns))],
        key=lambda x: abs(x[2]), reverse=True
    )
    
    total_mass = sum([abs(v) for _, _, v in feature_data]) + 1e-9
    data_lines, shap_lines = [], []
    
    for name, real_val, shap_val in feature_data[:3]:
        val_str = f"${real_val:,.2f}" if "Amount" in name or "Balance" in name or "Discrepancy" in name else f"{real_val:.2f}"
        hint = "ANOMALY (Increased Risk)" if shap_val > 0 else "CONSISTENT BEHAVIOR (Mitigated Risk)"
        data_lines.append(f"- {name}: {val_str}")
        shap_lines.append(f"- {name}: {hint} | Contribution: {(abs(shap_val)/total_mass)*100:.1f}%")

    prompt = f"""You are a Senior Model Risk Examiner. Write a strict, short compliance explanation.
    CONTEXT:\n{chr(10).join(data_lines)}\nRISK FACTORS:\n{chr(10).join(shap_lines)}
    Write a "Notice of Adverse Action" explanation. Use the provided logic hints. Interpret negative SHAP as consistency. Keep under 150 words. Professional tone only."""
    
    try:
        return client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=300
        ).choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"