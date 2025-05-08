# rag_service/tools/retriever.py
import os
import time
import httpx
from dotenv import load_dotenv
import logging
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Configuration
PERSIST_DIR = "./chroma_db"
EMBEDDING_CACHE_DIR = "./embedding_cache"
# COLLECTION_NAME = "lilian-blog"
COLLECTION_NAME = "uploaded-docs"
# go to container and pull the model fist eg. ollama pull mxbai-embed-large
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OLLAMA_SERVER_URL = os.getenv("BASE_URL")
BLOG_URLS = [
    "https://en.wikipedia.org/wiki/Indian_Penal_Code",
]
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
k = 3


def wait_for_ollama_model(model: str, base_url: str, timeout: int = 60, interval: int = 5):
    """Wait until the embedding model is available on Ollama."""
    embed_url = f"{base_url}/api/embed"
    dummy_payload = {"model": model, "prompt": "ping"}
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.post(embed_url, json=dummy_payload)
            if response.status_code != 404:
                logging.info(f"Ollama model '{model}' is available.")
                return True
        except Exception as e:
            logging.warning(f"Ollama not ready yet: {e}")
        logging.info(f"Waiting for Ollama model '{model}' to load...")
        time.sleep(interval)
    
    raise RuntimeError(f"Timeout waiting for Ollama model '{model}' to load.")


def is_chroma_db_initialized(persist_dir: str) -> list[str]:
    """Check if ChromaDB is properly initialized after saving."""
    db_file = os.path.join(persist_dir, 'chroma.sqlite3')
    if not os.path.exists(db_file):
        logging.info(f"ChromaDB file not found at {db_file}.")
        return []

    # Try to find a subdirectory that looks like a Chroma collection directory
    collection_dirs = [
        d for d in os.listdir(persist_dir) if os.path.isdir(os.path.join(persist_dir, d)) and
        len(d) == 36 and all(c in '0123456789abcdef-' for c in d)
    ]

    if not collection_dirs:
        logging.info("No Chroma collection directories found.")
        return []

    # Check for required files in each collection
    required_collection_files = ['header.bin', 'length.bin', 'link_lists.bin']
    valid_collections = []
    for collection_dir in collection_dirs:
        collection_path = os.path.join(persist_dir, collection_dir)
        if all(os.path.exists(os.path.join(collection_path, f)) for f in required_collection_files):
            valid_collections.append(collection_dir)

    return valid_collections

def initialize_retriever():
    # Wait for the embedding model to be available
    wait_for_ollama_model(EMBEDDING_MODEL, OLLAMA_SERVER_URL)

    # Configure Ollama embeddings with correct server URL
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_SERVER_URL
    )
    
    # Create directory if it doesn't exist
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    # Create cached embeddings
    fs = LocalFileStore(EMBEDDING_CACHE_DIR)
    logging.info(f"Creating cached embeddings at {EMBEDDING_CACHE_DIR}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        fs,
        namespace=EMBEDDING_MODEL
    )

    # Initialize ChromaDB and check if collections exist
    collection_dirs = is_chroma_db_initialized(PERSIST_DIR)
    if collection_dirs:
        try:
            logging.info(f"Found collections: {collection_dirs}")
            # Iterate over all collections
            for collection_name in collection_dirs:
                try:
                    logging.info(f"Loading ChromaDB collection: {collection_name}")
                    vectorstore = Chroma(
                        collection_name=collection_name,
                        persist_directory=PERSIST_DIR,
                        embedding_function=cached_embeddings,  # Use cached embeddings
                    ).as_retriever(search_kwargs={"k": k})
                    
                    # You can return a list of retrievers for each collection or aggregate results.
                    # In this example, I am returning the first retriever found:
                    return vectorstore
                except Exception as e:
                    logging.error(f"Failed to load ChromaDB for collection '{collection_name}': {e}")
        except Exception as e:
            logging.error(f"Error loading ChromaDB collections: {e}")
    else:
        # Create new vectorstore if needed
        logging.info("Creating new ChromaDB vectorstore")
        try:
            loader = WebBaseLoader(BLOG_URLS)
            
            docs = loader.load()
            logging.info(f"Loaded {len(docs)} documents from URLs {BLOG_URLS}") 
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            splits = splitter.split_documents(docs)

            # for just testing splits
            for doc in splits[:2]:
                logging.info(f"Sample chunk: {doc.page_content[:100]}")
                logging.info(f"Embedding: {cached_embeddings.embed_query(doc.page_content[:100])}")

            # Create and return the vectorstore - persistence is automatic
            logging.info(f"Creating new ChromaDB vectorstore at {PERSIST_DIR} and cache embeddings at {cached_embeddings}")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=cached_embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR
            )

            logging.info(f"Created new ChromaDB vectorstore at {PERSIST_DIR}")
            return vectorstore.as_retriever(search_kwargs={"k": k})
        except Exception as e:
            logging.error(f"Failed to create new ChromaDB: {e}")
            raise

retriever = initialize_retriever()