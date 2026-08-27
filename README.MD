
# Credit-Scout: AI Risk Engine & Compliance Reporter

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Azure Container Apps](https://img.shields.io/badge/Azure-Container%2520Apps-0089D6?style=for-the-badge&logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Groq](https://img.shields.io/badge/Groq-Qwen%2FLlama-blueviolet?style=for-the-badge)](https://groq.com/)

**Live Azure Application:** [Launch Credit-Scout Live](https://credit-scout-app.purpledesert-dd796257.southafricanorth.azurecontainerapps.io/)  
**Hugging Face Spaces (Alternative):** [View HF Space](https://huggingface.co/spaces/oke39/Credit-Scout-Live)  
**Experiment Tracking:** [View Training Metrics on DagsHub](https://dagshub.com/oke03940/Credit-Scout-Fraud-Detection/experiments)

<img width="1895" height="907" alt="Image" src="https://github.com/user-attachments/assets/ed767f7e-05ad-438b-ae14-2a085e7ff93b" />

## Overview
Credit-Scout is a production-grade, real-time financial fraud detection and compliance platform that bridges the gap between **Deep Learning AI** and **Regulatory Compliance**. 

It combines an **LSTM neural network** for real-time transaction scoring, **SHAP (SHapley Additive exPlanations)** for explainable AI, and **Groq-powered LLMs** to automatically generate audit-ready "Notices of Adverse Action"—all backed by a persistent PostgreSQL audit trail.

## Architecture & Tech Stack
1. **API Layer:** FastAPI with asynchronous startup lifespan handling and static frontend routing (`main.py`, `schemas.py`).
2. **ML Runtime:** TensorFlow/Keras LSTM model combined with automated SHAP attribution generation (`ml_service.py`).
3. **Compliance Engine:** Groq API integration (Qwen/Llama models) to parse risk attributions into natural language regulatory reports.
4. **Data Persistence:** SQLAlchemy ORM coupled with a secure PostgreSQL database (`models.py`, `database.py`) logging every transaction review.
5. **Cloud Deployment:** Containerized via Docker (`python:3.9-slim`) and orchestrated natively on **Azure Container Apps** with an optimized **~630ms average end-to-end response latency**.

## Features
* **Real-time Scoring:** Instant probability assessment of transfer and cash-out transactions with live visual gauge rendering.
* **Explainable AI (XAI):** Feature attribution parsing that highlights transaction anomalies and risk weights.
* **Automated Audit Reporting:** Generates regulatory-compliant text explanations for flagged financial decisions.
* **Persistent Transaction Logging:** Automatically commits every transaction scan, risk score, and LLM report to a secure PostgreSQL database for audit tracking.

## Installation & Local Setup

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

* `ml_service.py`: Core machine learning inference, feature scaling, SHAP computation, and Groq LLM reporting.

* `models.py`: SQLAlchemy database models for transaction audit logging.

* `schemas.py`: Pydantic request and response validation models.

* `database.py`: PostgreSQL database engine configuration.

* `static/`: Production HTML/CSS frontend dashboard interface.

* `latest_checkpoint.h5`: Serialized LSTM model weights.

* `scaler.pkl & columns.pkl`: Feature preprocessing and scaling artifacts.

---

*Created by Oke Oluwanifemi - 2026*

```

