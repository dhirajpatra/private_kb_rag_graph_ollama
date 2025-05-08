# rag_service/app.py
import os
from typing import Optional
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from graph.rag_graph import RAGGraphService
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

# @app.post("/process-to-kg")
# async def process_to_kg(file: UploadFile, source_type: str):
#     """Unified endpoint for KG processing, saving, and graph creation"""
#     try:
#         # Save temp file
#         file_path = f"/tmp/{file.filename}"
#         with open(file_path, "wb") as f:
#             f.write(await file.read())

#         # Unified KG workflow
#         result = await kg_builder.process_document(file_path, source_type)
#         return {
#             "status": "success",
#             "message": "Knowledge Graph processed successfully.",
#             "details": result
#         }
#     except Exception as e:
#         logging.exception("KG processing failed.")
#         raise HTTPException(status_code=500, detail=str(e))


#     # Process through KG builder
#     result = await kg_builder.process_document(file_path, source_type)
#     return result

# @app.get("/ask")
# async def ask_question(question: str, mode: str = "vector"):
#     """Query the knowledge graph"""
#     return await kg_builder.query_graph(question, mode)

# pdf upload and vectorize for RAG
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...), collection_name: Optional[str] = "uploaded-docs"):
    try:
        from tools.uploader import save_uploaded_pdf_to_chroma
        
        # Save the uploaded PDF to Chroma
        pdf_path = save_uploaded_pdf_to_chroma(file, collection_name)
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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/runs/batch")
def run_batch():
    return {"status": "OK", "message": "Batch endpoint placeholder"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

