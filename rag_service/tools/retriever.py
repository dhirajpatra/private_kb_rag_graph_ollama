# rag_service/tools/retriever.py
import os
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
    "https://developers.google.com/machine-learning/resources/prompt-eng",
]
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
k = 3

def is_chroma_db_initialized(persist_dir: str) -> bool:
    """Check if ChromaDB is properly initialized after saving."""
    db_file = os.path.join(persist_dir, 'chroma.sqlite3')
    if not os.path.exists(db_file):
        logging.info(f"ChromaDB file not found in retriever tool at {db_file}.")
        return False

    # Try to find a subdirectory that looks like a Chroma collection directory
    collection_dirs = [
        d for d in os.listdir(persist_dir) if os.path.isdir(os.path.join(persist_dir, d)) and
        len(d) == 36 and all(c in '0123456789abcdef-' for c in d)
    ]

    if not collection_dirs:
        logging.info("No Chroma collection directories found.")
        return False

    # Check for the existence of essential files within the first collection directory found
    collection_path = os.path.join(persist_dir, collection_dirs[0])
    required_collection_files = ['header.bin', 'length.bin', 'link_lists.bin']
    return all(os.path.exists(os.path.join(collection_path, f)) for f in required_collection_files)

def initialize_retriever():
    # Configure Ollama embeddings with correct server URL
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_SERVER_URL
    )
    
    # Create directory if it doesn't exist
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    # Create cached embeddings
    fs = LocalFileStore(EMBEDDING_CACHE_DIR)
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
    
    # Create new vectorstore if needed
    logging.info("Creating new ChromaDB in retrierver tool at vectorstore")
    try:
        loader = WebBaseLoader(BLOG_URLS)
        
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = splitter.split_documents(docs)
        logging.info(f"Number of splits: {len(splits)}")
        # Create and return the vectorstore - persistence is automatic
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=cached_embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR
        )
        logging.info(f"Vectorstore created successfully: {vectorstore}")
        return vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logging.error(f"Failed to create new ChromaDB: {e}")
        raise

retriever = initialize_retriever()