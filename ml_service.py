import os 
import numpy as np
import tensorflow as tf 
import shap 
import pickle 
from groq import Groq 

# Initialize global resources
model, scaler, columns, explainer = None, None , None ,None
groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

BUSSINESS_MAP = {
    'step': 'Transaction Hour', 'type_enc': 'Txn Type (Transfer/CashOut)',
    'amount': 'Transaction Amount', 'oldbalanceOrg': 'Origin Acct Balance (Pre)',
    'newbalanceOrig': 'Origin Acct Balance (Post)', 'oldbalanceDest': 'Recipient Acct Balance (Pre)',
    'newbalanceDest': 'Recipient Acct Balance (Post)', 'errorBalanceOrig': 'Origin Math Discrepancy',
    'errorBalanceDest': 'Recipient Math Discrepancy'
}


def load_ml_resources():
    global model,scaler,columns,explainer
    model = tf.keras.models.load_model('latest_checkpoint.h5')
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f: columns = pickle.load(f)
    with open('shap_metadata.pkl', 'rb') as f: shap_data = pickle.load(f)
    explainer = shap.GradientExplainer(model, shap_data['background_sample'])


def process_transaction(req):
    type_val = 0 if req.txn_type == 'TRANSFER' else 1
    error_bal_orig = req.new_balance_orig + req.amount - req.old_balance_org
    
    raw_features = np.array([
        150, type_val, req.amount, req.old_balance_org, req.new_balance_orig, 
        0.0, 0.0, error_bal_orig, 0.0
    ]).reshape(1, -1)
    
    scaled = scaler.transform(raw_features)
    lstm_input = scaled.reshape(1, 1, 9)
    
    risk_prob = float(model.predict(lstm_input)[0][0])
    shap_vals = explainer.shap_values(lstm_input)
    
    report = generate_llm_report(shap_vals, lstm_input, raw_features)
    return risk_prob, report, error_bal_orig


def generate_llm_report(shap_values, lstm_input, raw_features):
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
        val_str = f"${real_val:,.2f}" if "Amount" in name or "Balance" in name else f"{real_val:.2f}"
        hint = "ANOMALY (Increased Risk)" if shap_val > 0 else "CONSISTENT BEHAVIOR (Mitigated Risk)"
        data_lines.append(f"- {name}: {val_str}")
        shap_lines.append(f"- {name}: {hint} | Contribution: {(abs(shap_val)/total_mass)*100:.1f}%")

    prompt = f"""You are a Senior Model Risk Examiner. Write a short compliance explanation.
    CONTEXT:\n{chr(10).join(data_lines)}\nRISK FACTORS:\n{chr(10).join(shap_lines)}
    Write a "Notice of Adverse Action" explanation. Use logic hints. Keep under 150 words."""
    
    try:
        return client.chat.completions.create(
            model="llama-3.1-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=300
        ).choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"