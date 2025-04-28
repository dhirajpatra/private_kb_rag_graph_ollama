# rag_service/tools/retriever.py
import os
import bs4
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
COLLECTION_NAME = "lilian-blog"
BLOG_URLS = [
    "https://developers.google.com/machine-learning/resources/prompt-eng",
]
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
k = 3

# Check if ChromaDB is initialized
def is_chroma_db_initialized(persist_dir: str) -> bool:
    """Check if ChromaDB is properly initialized after saving."""
    db_file = os.path.join(persist_dir, 'chroma.sqlite3')
    if not os.path.exists(db_file):
        logging.info(f"ChromaDB file not found at {db_file}.")
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

# Initialize retriever and load or create ChromaDB
def initialize_retriever():
    # Configure Ollama embeddings with correct server URL
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        base_url=os.getenv("BASE_URL")
    )
    
    # Create directory if it doesn't exist
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    # Create cached embeddings
    fs = LocalFileStore(EMBEDDING_CACHE_DIR)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        fs,
        namespace=os.getenv("EMBEDDING_MODEL")
    )

    # Check if ChromaDB is initialized and load it
    if is_chroma_db_initialized(PERSIST_DIR):
        try:
            logging.info("Loading existing ChromaDB from persistence directory")
            
            # Dynamically detect all collections
            collection_dirs = [
                d for d in os.listdir(PERSIST_DIR) if os.path.isdir(os.path.join(PERSIST_DIR, d)) and
                len(d) == 36 and all(c in '0123456789abcdef-' for c in d)
            ]
            
            retrievers = []
            for collection_id in collection_dirs:
                retrievers.append(
                    Chroma(
                        collection_name=collection_id,  # Collection name = folder ID
                        persist_directory=PERSIST_DIR,
                        embedding_function=cached_embeddings,
                    ).as_retriever(search_kwargs={"k": k})
                )

            # Custom retriever to search all collections
            class MultiRetriever:
                def __init__(self, retrievers):
                    self.retrievers = retrievers
                    self.search_kwargs = {"k": k}

                def invoke(self, query):
                    results = []
                    for r in self.retrievers:
                        results.extend(r.invoke(query))
                    return results
            
            return MultiRetriever(retrievers)
        
        except Exception as e:
            logging.error(f"Failed to load existing ChromaDB: {e}")
            raise

    # Create new vectorstore if needed
    logging.info("Creating new ChromaDB vectorstore")
    try:
        loader = WebBaseLoader(
            web_paths=(BLOG_URLS,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )

        # load web page
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = splitter.split_documents(docs)
        
        # Create and return the vectorstore - persistence is automatic
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=cached_embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR
        )
        
        return vectorstore.as_retriever(search_kwargs={"k": k})
    
    except Exception as e:
        logging.error(f"Failed to create new ChromaDB: {e}")
        raise

# Initialize retriever
retriever = initialize_retriever()
