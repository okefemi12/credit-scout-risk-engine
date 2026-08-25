# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import database, models, schemas, ml_service
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts into memory on startup
    ml_service.load_ml_resources()
    # Create database tables
    models.Base.metadata.create_all(bind=database.engine)
    yield

app = FastAPI(title="Credit-Scout Risk API", lifespan=lifespan)


app.mount("/static", StaticFiles(directory="static"), name="static") 


# Serve the dashboard on the root URL
@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")  

@app.post("/analyze-risk", response_model=schemas.TransactionResponse)
def analyze_risk(req: schemas.TransactionRequest, db: Session = Depends(database.get_db)):
    try:
        # Run ML Pipeline
        risk_score, llm_report, error_bal = ml_service.process_transaction(req)
        status = "FLAGGED" if risk_score > 0.8 else "APPROVED"
        
        # Log to Database
        audit_log = models.CreditTransactionAudit(
            txn_type=req.txn_type,
            amount=req.amount,
            old_balance_org=req.old_balance_org,
            new_balance_orig=req.new_balance_orig,
            risk_score=risk_score,
            llm_explanation=llm_report,
            status=status
        )
        db.add(audit_log)
        db.commit()
        
        return schemas.TransactionResponse(
            risk_score=risk_score, 
            status=status, 
            llm_explanation=llm_report,
            math_discrepancy=error_bal
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))