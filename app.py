import streamlit as st
import numpy as np
import tensorflow as tf
import shap
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq 

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Credit-Scout AI", layout="wide")

# CSS for "Bank" styling
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #000044; color: white; width: 100%; }
    .risk-high { color: #cc0000; font-weight: bold; font-size: 20px; }
    .risk-low { color: #006600; font-weight: bold; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

# Initialize Groq Client
# It looks for the key in Hugging Face Secrets
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("⚠️ GROQ_API_KEY not found in Secrets! The LLM explanation will fail.")
    client = None
else:
    client = Groq(api_key=api_key)

# --- 2. LOAD ARTIFACTS ---
@st.cache_resource
def load_resources():
    # Load Model (CPU mode)
    model = tf.keras.models.load_model('latest_checkpoint.h5')
    
    # Load Pickles
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    with open('shap_metadata.pkl', 'rb') as f:
        shap_data = pickle.load(f)
        
    # Re-initialize Explainer
    explainer = shap.GradientExplainer(model, shap_data['background_sample'])
    return model, scaler, columns, explainer

# Load everything once
try:
    model, scaler, columns, explainer = load_resources()
except Exception as e:
    st.error(f"Error loading files: {e}. Did you upload .h5 and .pkl files?")
    st.stop()

# --- 3. BUSINESS MAPPING ---
BUSINESS_MAP = {
    'step': 'Transaction Hour',
    'type_enc': 'Txn Type (Transfer/CashOut)',
    'amount': 'Transaction Amount',
    'oldbalanceOrg': 'Origin Acct Balance (Pre)',
    'newbalanceOrig': 'Origin Acct Balance (Post)',
    'oldbalanceDest': 'Recipient Acct Balance (Pre)',
    'newbalanceDest': 'Recipient Acct Balance (Post)',
    'errorBalanceOrig': 'Origin Math Discrepancy',
    'errorBalanceDest': 'Recipient Math Discrepancy'
}

# --- 4. EXPLANATION FUNCTION (GROQ API) ---
def generate_explanation_cloud(sample_idx_in_shap, shap_values, original_samples, feature_names, scaler):
    # A. Inverse Transform to get Real Money
    raw_scaled = original_samples.flatten()
    real_values = scaler.inverse_transform(raw_scaled.reshape(1, -1)).flatten()
    
    if isinstance(shap_values, list):
        vals = shap_values[0]
    else:
        vals = shap_values
    vals = vals.flatten()
    
    # B. Prepare Data for LLM
    feature_data = []
    for i, col_name in enumerate(feature_names):
        biz_name = BUSINESS_MAP.get(col_name, col_name)
        feature_data.append((biz_name, real_values[i], vals[i]))
    
    # Sort by absolute impact
    feature_data.sort(key=lambda x: abs(x[2]), reverse=True)
    total_shap_mass = sum([abs(v) for _, _, v in feature_data]) + 1e-9
    
    data_lines = []
    shap_lines = []
    
    for name, real_val, shap_val in feature_data[:3]:
        # Format Currency
        if "Amount" in name or "Balance" in name:
            val_str = f"${real_val:,.2f}"
        else:
            val_str = f"{real_val:.2f}"
            
        contrib_pct = (abs(shap_val) / total_shap_mass) * 100
        logic_hint = "ANOMALY (Increased Risk)" if shap_val > 0 else "CONSISTENT BEHAVIOR (Mitigated Risk)"
        
        data_lines.append(f"- {name}: {val_str}")
        shap_lines.append(f"- {name}: {logic_hint} | Contribution: {contrib_pct:.1f}%")

    # C. Call Groq
    if not client:
        return "Error: Groq API Key missing."

    prompt = f"""
    You are a Senior Model Risk Examiner. Write a strict, short compliance explanation.
    
    CONTEXT:
    {chr(10).join(data_lines)}
    
    RISK FACTORS:
    {chr(10).join(shap_lines)}
    
    Write a "Notice of Adverse Action" explanation. 
    Use the provided logic hints. Interpret negative SHAP as consistency. 
    Keep it under 150 words. Professional tone only. Add a standard disclaimer at the end.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

# --- 5. SIDEBAR UI ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=100)
st.sidebar.title("💳 Transaction Details")

amount = st.sidebar.number_input("Amount ($)", value=350000.0)
old_bal = st.sidebar.number_input("Origin Old Balance ($)", value=1200000.0)
new_bal = st.sidebar.number_input("Origin New Balance ($)", value=850000.0)
txn_type = st.sidebar.selectbox("Type", ["TRANSFER", "CASH_OUT"])

# Auto-calculate math features
error_bal_orig = new_bal + amount - old_bal
st.sidebar.info(f"Math Discrepancy: ${error_bal_orig:.2f}")

# --- 6. MAIN APP LOGIC ---
st.title("🏦 Credit-Scout AI Risk Engine")
st.markdown("Real-time Fraud Detection with Llama 3.1 Explainability")

if st.sidebar.button("Analyze Transaction"):
    with st.spinner("Analyzing Risk Patterns..."):
        # 1. Preprocess
        type_val = 0 if txn_type == 'TRANSFER' else 1
        
        # Construct Input Array (Must match columns.pkl order exactly!)
        # Standard PaySim columns: step, type, amount, oldBalOrg, newBalOrig, oldBalDest, newBalDest, errorOrig, errorDest
        raw_features = np.array([
            150,        # step (mock)
            type_val,
            amount,
            old_bal,
            new_bal,
            0.0,        # oldbalanceDest (mock)
            0.0,        # newbalanceDest (mock)
            error_bal_orig,
            0.0         # errorBalanceDest (mock)
        ]).reshape(1, -1)
        
        # Scale & Reshape
        scaled_features = scaler.transform(raw_features)
        lstm_input = scaled_features.reshape(1, 1, 9)
        
        # 2. Predict
        risk_prob = model.predict(lstm_input)[0][0]
        
        # 3. Explain (SHAP)
        shap_vals = explainer.shap_values(lstm_input)
        
        # 4. Display Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Risk Score")
            st.metric(label="Fraud Probability", value=f"{risk_prob:.2%}")
            if risk_prob > 0.8:
                st.markdown('<p class="risk-high">⛔ FLAGGED</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="risk-low">✅ APPROVED</p>', unsafe_allow_html=True)
                
        with col2:
            st.subheader("Model Logic (SHAP)")
            # Fix SHAP plot dimensions
            st.set_option('deprecation.showPyplotGlobalUse', False)
            if isinstance(shap_vals, list):
                shap_vals_plot = shap_vals[0]
            else:
                shap_vals_plot = shap_vals
            
            fig = plt.figure()
            shap.summary_plot(shap_vals_plot, raw_features, feature_names=columns, plot_type="bar", show=False)
            st.pyplot(fig)
            
        # 5. LLM Report
        st.markdown("---")
        st.subheader("📝 Audit Report (Llama 3.1)")
        with st.spinner("Drafting Compliance Notice via Groq..."):
            report = generate_explanation_cloud(0, shap_vals, lstm_input, columns, scaler)
            st.success("Report Generated")
            st.write(report)

else:
    st.info(" Adjust transaction details in the sidebar to test the model.")