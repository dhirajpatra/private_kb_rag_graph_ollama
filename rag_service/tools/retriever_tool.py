# tools/retriever_tool.py
import logging
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tools.retriever import retriever  # This imports the pre-initialized retriever
from functools import lru_cache  # Import LRU cache

logging.basicConfig(level=logging.INFO)
k = 3  # Default number of documents to return

class RetrieverToolArgs(BaseModel):
    query: str = Field(description="The search query to retrieve information from vector databases.")
    k: int = Field(default=k, description="Number of documents to return")

@tool(args_schema=RetrieverToolArgs)
@lru_cache(maxsize=100)  # Cache up to 100 results
def retriever_tool(query: str, k: int = 1) -> dict:
    """
    Search RAG vector databases.
    Returns relevant passages based on semantic similarity.
    """
    try:
        # Update search parameters
        retriever.search_kwargs["k"] = k
        results = retriever.invoke(query)
        logging.info(f"[rag_retriever] Retrieved {len(results)} documents")
        return {
            "status": "success",
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        }
    except Exception as e:
        logging.error(f"[rag_retriever] Error: {e}")
        return {"status": "error", "message": f"Retrieval failed: {str(e)}"}
