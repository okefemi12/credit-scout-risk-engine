from pydantic import BaseModel

class TransactionRequest(BaseModel):
    amount: float
    old_balance_org: float
    new_balance_orig: float
    txn_type: str  # "TRANSFER" or "CASH_OUT"

class TransactionResponse(BaseModel):
    risk_score: float
    status: str
    llm_explanation: str
    math_discrepancy: float