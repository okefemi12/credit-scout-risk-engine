# models.py
from sqlalchemy import Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class CreditTransactionAudit(Base):
    __tablename__ = "transaction_audits"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Input Features
    txn_type = Column(String(50), index=True)
    amount = Column(Float)
    old_balance_org = Column(Float)
    new_balance_orig = Column(Float)
    error_balance_orig = Column(Float)

    # ML & AI Outputs
    risk_score = Column(Float)  # The fraud probability from the LSTM
    shap_top_feature = Column(String(100)) # e.g., "Transaction Amount: ANOMALY"
    llm_explanation = Column(Text) # The Notice of Adverse Action from Groq

    # System Tracking
    status = Column(String(20)) # "FLAGGED" or "APPROVED"