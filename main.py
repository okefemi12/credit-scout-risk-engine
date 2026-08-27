# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import database, models, schemas, ml_service
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Starting up: Loading ML resources...")
        ml_service.load_ml_resources()  
        print("Starting up: Creating database tables...")
        models.Base.metadata.create_all(bind=database.engine)
        print("Startup complete successfully!")
    except Exception as e:
        print(f"CRITICAL STARTUP ERROR: {str(e)}")
        raise e  # This ensures the container fails visibly if models are missing
    yield

app = FastAPI(title="Credit-Scout Risk API", lifespan=lifespan)


app.mount("/static", StaticFiles(directory="static"), name="static") 


# Serve the dashboard on the root URL
@app.get("/")
def serve_dashboard():
    file_path = "static/index.html"
    if not os.path.exists(file_path):
        # List what files actually exist in the container to debug
        current_dir_contents = os.listdir(".")
        return JSONResponse(
            status_code=404,
            content={
                "error": "index.html not found",
                "looking_at_path": os.path.abspath(file_path),
                "root_directory_contents": current_dir_contents
            }
        )
    return FileResponse(file_path)

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