
# Credit-Scout: AI Risk Engine & Compliance Reporter

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://huggingface.co/spaces/oke39/Credit-Scout-Live)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://dagshub.com/oke03940/Credit-Scout-Fraud-Detection/experiments)
[![Groq](https://img.shields.io/badge/Groq-Llama3-blueviolet?style=for-the-badge)](https://groq.com/)

**Live Demo:** [Click Here to Launch App](https://huggingface.co/spaces/oke39/Credit-Scout-Live)  
**Experiment Tracking:** [View Training Metrics on DagsHub](https://dagshub.com/oke03940/Credit-Scout-Fraud-Detection/experiments)
<img width="1895" height="907" alt="Image" src="https://github.com/user-attachments/assets/ed767f7e-05ad-438b-ae14-2a085e7ff93b" />
##  Overview
Credit-Scout is a real-time financial fraud detection system that bridges the gap between **Black-Box AI** and **Regulatory Compliance**. 

It uses an **LSTM (Long Short-Term Memory)** neural network to detect fraudulent transaction patterns and generates human-readable "Adverse Action Notices" using **Llama 3.3 (via Groq)**.

## Architecture
1.  **Detection Layer:** LSTM Model trained on the PaySim dataset (TensorFlow/Keras).
2.  **Explainability Layer:** SHAP (SHapley Additive exPlanations) to isolate key risk factors.
3.  **Compliance Layer:** Llama 3.3 generates a natural language audit report explaining *why* a transaction was flagged.
4.  **Deployment:** Dockerized Streamlit app hosted on Hugging Face Spaces.

## Features
* **Real-time Scoring:** Instant probability assessment of transaction requests.
* **Visual Forensics:** SHAP Waterfall/Bar plots showing feature contributions.
* **AI Compliance Officer:** Generates automated text reports for non-technical stakeholders.

##  Installation & Setup
To run this locally:

1. **Clone the repo**
   ```bash
   git clone [https://github.com/oke39/credit-scout-risk-engine.git](https://github.com/oke39/credit-scout-risk-engine.git)
   cd credit-scout-risk-engine

```

2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Set up API Keys**
Create a `.env` file or export your Groq key:
```bash
export GROQ_API_KEY="gsk_..."

```


4. **Run the App**
```bash
streamlit run app.py

```



## 📂 Project Structure

* `app.py`: Main application logic.
* `models/`: Serialized model artifacts (Scaler, LSTM, SHAP Explainers).
* `notebooks/`: Model training and evaluation notebooks.

---

*Created by [oke39] - 2026*

```

