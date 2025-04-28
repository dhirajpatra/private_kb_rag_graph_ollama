# tools/uploader.py
import os
import requests
import logging
from pathlib import Path
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

def save_drive_pdf_to_chroma(drive_url: str, collection_name: str):
    # Create temp dir
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Download the file
    file_name = drive_url.split("/")[-2] + ".pdf"  # Rough parsing
    local_path = temp_dir / file_name

    response = requests.get(drive_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")
    
    with open(local_path, "wb") as f:
        f.write(response.content)

    # embeddings
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_SERVER_URL
    )

    # create folders if not exist
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    fs = LocalFileStore(EMBEDDING_CACHE_DIR)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        fs,
        namespace=EMBEDDING_MODEL
    )

    # load PDF
    loader = PyPDFLoader(local_path)
    docs = loader.load()

    # split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = splitter.split_documents(docs)

    # save into Chroma
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=cached_embeddings,
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
    )

    # Do not return vectorstore
    return {
        "status": "OK",
        "message": f"Your file vectorized and saved into vector database {file_name}"
    }
