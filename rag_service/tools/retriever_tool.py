# tools/retriever_tool.py
import logging
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tools.retriever import retriever  # This imports the pre-initialized retriever

logging.basicConfig(level=logging.INFO)
k = 3  # Default number of documents to return

class RetrieverToolArgs(BaseModel):
    query: str = Field(description="The search query to retrive information in the blog posts.")
    k: int = Field(default=k, description="Number of documents to return")

@tool(args_schema=RetrieverToolArgs)
def retriever_tool(query: str, k: int = 1) -> dict:
    """
    Search blog posts about LLM, prompt engineering, and adversarial attacks.
    Returns relevant passages based on semantic similarity.
    """
    logging.info(f"*********************** [blog_retriever] Searching for: {query}")
    try:
        # Update search parameters
        retriever.search_kwargs["k"] = k
        results = retriever.invoke(query)
        logging.info(f"[blog_retriever] Retrieved {len(results)} documents")
        return {
            "status": "success",
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        }
    except Exception as e:
        logging.error(f"[blog_retriever] Error: {e}")
        return {"status": "error", "message": f"Retrieval failed: {str(e)}"}