# rag_service/app.py
import os
from typing import Optional
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from graph.rag_graph import RAGGraphService
from tools.uploader import save_uploaded_pdf_to_chroma
# from slowapi import Limiter
# from slowapi.util import get_remote_address

# Load env variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

# FastAPI app
# limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGGraphService()

# Request/Response Models
class InputMessage(BaseModel):
    text: str

class OutputMessage(BaseModel):
    reply: str

# pdf upload and vectorize for RAG
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), collection_name: Optional[str] = "uploaded-docs"):
    try:
        # Save the uploaded PDF to Chroma
        pdf_path = await save_uploaded_pdf_to_chroma(file, collection_name)
        return {"status": "success", "message": f"PDF uploaded and vectorized.", "local_path": pdf_path}
    except Exception as e:
        logging.exception("Failed to process uploaded PDF.")
        raise HTTPException(status_code=500, detail=str(e))
    
# Chat endpoint
@app.post("/chat", response_model=OutputMessage)
# @limiter.limit("1/minute")
def chat(input_msg: InputMessage):
    try:
        logging.info(f"Received message: {input_msg.text}")
        reply = rag_service.run(query=input_msg.text)  # Ensure this is a valid string or dict
        return {"reply": reply}
    except Exception as e:
        logging.exception("Chat processing failed.")
        raise HTTPException(status_code=500, detail="Unexpected server error.")

@app.get("/")
def health():
    return {"status": "agent running"}

@app.get("/info")
def info():
    return {"status": "ok"}

@app.post("/runs/batch")
def run_batch():
    return {"status": "OK", "message": "Batch endpoint placeholder"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

