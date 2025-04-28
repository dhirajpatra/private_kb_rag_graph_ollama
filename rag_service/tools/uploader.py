# tools/uploader.py
import os
import logging
import shutil
import tempfile
from pathlib import Path
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

PERSIST_DIR = "./chroma_db"
EMBEDDING_CACHE_DIR = "./embedding_cache"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OLLAMA_SERVER_URL = os.getenv("BASE_URL")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
k = 3

def save_uploaded_pdf_to_chroma(file: UploadFile, collection_name: str):
    try:
        """Check if ChromaDB is properly initialized after saving."""
        db_file = os.path.join(PERSIST_DIR, 'chroma.sqlite3')
        if not os.path.exists(db_file):
            logging.info(f"ChromaDB file not found at {db_file}.")
            return False
    
        # Create temp directory for the uploaded file
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save the uploaded PDF to the temporary directory
        local_path = temp_dir / file.filename
        with open(local_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Embedding setup
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_SERVER_URL
        )

        # Ensure persistence directory exists
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

        # File system for cache-backed embeddings
        fs = LocalFileStore(EMBEDDING_CACHE_DIR)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            fs,
            namespace=EMBEDDING_MODEL
        )

        # Load the PDF file using PyPDFLoader
        loader = PyPDFLoader(local_path)
        docs = loader.load()

        # Split the documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = splitter.split_documents(docs)

        # Save the document splits into Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=cached_embeddings,
            collection_name=collection_name,
            persist_directory=PERSIST_DIR,
        )

        # Return success message
        return {
            "status": "OK",
            "message": f"Your file '{file.filename}' has been vectorized and saved into the vector database."
        }
    
    except Exception as e:
        logging.error(f"Error processing the uploaded PDF: {e}")
        raise Exception(f"Error processing the file: {e}")
