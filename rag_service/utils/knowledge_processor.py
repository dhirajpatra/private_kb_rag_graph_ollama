# rag_service/utils/knowledge_processor.py

import os
import logging
from typing import List, Dict
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from ..knowledge_graph.src.shared.common_fn import (
    check_url_source,
    load_embedding_model,
    create_graph_database_connection,
    handle_backticks_nodes_relationship_id_type,
    save_graphDocuments_in_neo4j,
    execute_graph_query
)

class KnowledgeProcessor:
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base

    def process_knowledge(self, query: str) -> str:
        return f"Processed knowledge for query: {query}"

    @staticmethod
    def check_url_source_api(request: Dict) -> Dict:
        source_type = request.get("source_type")
        yt_url = request.get("yt_url", "")
        wiki_query = request.get("wiki_query", "")
        result = check_url_source(source_type, yt_url, wiki_query)
        return {"result": result}

    @staticmethod
    def load_embedding_model_api(request: Dict) -> Dict:
        model_name = request.get("model_name", "huggingface")
        embeddings, dim = load_embedding_model(model_name)
        return {"embedding_model": str(type(embeddings)), "dimension": dim}

    @staticmethod
    def create_graph_connection_api(request: Dict) -> Dict:
        graph = create_graph_database_connection(
            uri=request["uri"],
            userName=request["userName"],
            password=request["password"],
            database=request["database"]
        )
        return {"graph": graph}

    @staticmethod
    def save_graph_documents_api(graph: Neo4jGraph, graph_documents: List[GraphDocument]) -> Dict:
        cleaned_docs = handle_backticks_nodes_relationship_id_type(graph_documents)
        save_graphDocuments_in_neo4j(graph, cleaned_docs)
        return {"message": "Documents saved"}

    @staticmethod
    def query_graph_api(request: Dict) -> Dict:
        graph = request["graph"]
        query = request["query"]
        params = request.get("params", None)
        result = execute_graph_query(graph, query, params)
        return {"result": result}
