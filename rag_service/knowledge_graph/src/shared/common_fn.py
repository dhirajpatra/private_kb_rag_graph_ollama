import hashlib
import logging
from ..document_sources.youtube import create_youtube_url
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from neo4j.exceptions import TransientError
from langchain_community.graphs.graph_document import GraphDocument
from typing import List
import re
import os
import time
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlparse
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Configuration
PERSIST_DIR = "./chroma_db"
EMBEDDING_CACHE_DIR = "./embedding_cache"
# COLLECTION_NAME = "lilian-blog"
COLLECTION_NAME = "uploaded-docs"
# go to container and pull the model fist eg. ollama pull mxbai-embed-large
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_MODEL_DIMENSION = 1024
OLLAMA_SERVER_URL = os.getenv("BASE_URL")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
k = 3

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
MODEL=os.getenv("MODEL")

def check_url_source(source_type, yt_url:str=None, wiki_query:str=None):
    language=''
    try:
      logging.info(f"incoming URL: {yt_url}")
      if source_type == 'youtube':
        if re.match('(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/?.*(?:watch|embed)?(?:.*v=|v\/|\/)([\w\-_]+)\&?',yt_url.strip()):
          youtube_url = create_youtube_url(yt_url.strip())
          logging.info(youtube_url)
          return youtube_url,language
        else:
          raise Exception('Incoming URL is not youtube URL')
      
      elif  source_type == 'Wikipedia':
        wiki_query_id=''
        #pattern = r"https?:\/\/([a-zA-Z0-9\.\,\_\-\/]+)\.wikipedia\.([a-zA-Z]{2,3})\/wiki\/([a-zA-Z0-9\.\,\_\-\/]+)"
        wikipedia_url_regex = r'https?:\/\/(www\.)?([a-zA-Z]{2,3})\.wikipedia\.org\/wiki\/(.*)'
        wiki_id_pattern = r'^[a-zA-Z0-9 _\-\.\,\:\(\)\[\]\{\}\/]*$'
        
        match = re.search(wikipedia_url_regex, wiki_query.strip())
        if match:
                language = match.group(2)
                wiki_query_id = match.group(3)
          # else : 
          #       languages.append("en")
          #       wiki_query_ids.append(wiki_url.strip())
        else:
            raise Exception(f'Not a valid wikipedia url: {wiki_query} ')

        logging.info(f"wikipedia query id = {wiki_query_id}")     
        return wiki_query_id, language     
    except Exception as e:
      logging.error(f"Error in recognize URL: {e}")
      raise Exception(e)


def get_chunk_and_graph_document(graph_documents, chunk_id_chunk_doc_list):
    logging.info("Creating list of chunks and graph documents in get_chunk_and_graph_document")

    return [
        {'graph_doc': graph_doc, 'chunk_id': chunk_id}
        for graph_doc in graph_documents
        if 'combined_chunk_ids' in graph_doc.source.metadata
        for chunk_id in graph_doc.source.metadata['combined_chunk_ids']
    ] 
                 
def create_graph_database_connection(uri, 
                                     userName, 
                                     password, 
                                     database):
    # Check if user agent customization is enabled via environment variable
    enable_user_agent = os.environ.get("ENABLE_USER_AGENT", "False").lower() in ("true", "1", "yes")

    # If enabled, create Neo4jGraph connection with a custom user agent from env
    if enable_user_agent:
        graph = Neo4jGraph(
            url=uri,
            database=database,
            username=userName,
            password=password,
            refresh_schema=False,
            sanitize=True,
            driver_config={'user_agent': os.environ.get('NEO4J_USER_AGENT')}
        )
    else:
        # Otherwise, create Neo4jGraph connection without custom user agent
        graph = Neo4jGraph(
            url=uri,
            database=database,
            username=userName,
            password=password,
            refresh_schema=False,
            sanitize=True
        )
    
    # Return the connected Neo4j graph object
    return graph

def load_embedding_model(embedding_model_name: str = os.getenv("EMBEDDING_MODEL_SERVER")):
    if embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logging.info(f"Embedding: Using OpenAI Embeddings , Dimension:{dimension}")
    elif embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
          model=EMBEDDING_MODEL,
          base_url=OLLAMA_SERVER_URL
        )
        dimension = EMBEDDING_MODEL_DIMENSION
        logging.info(f"Embedding: Using Ollama Embeddings , Dimension:{dimension}")
    else:
        embeddings = OllamaEmbeddings(
            model_name=EMBEDDING_MODEL, #, cache_folder="/embedding_model"
            base_url=OLLAMA_SERVER_URL
        )
        dimension = 384
        logging.info(f"Embedding: Using Langchain OllamaEmbeddings , Dimension:{dimension}")
    return embeddings, dimension

def save_graphDocuments_in_neo4j(graph: Neo4jGraph, graph_document_list: List[GraphDocument], max_retries=3, delay=1):
   retries = 0
   while retries < max_retries:
       try:
           graph.add_graph_documents(graph_document_list, baseEntityLabel=True)
           return
       except TransientError as e:
           if "DeadlockDetected" in str(e):
               retries += 1
               logging.info(f"Deadlock detected. Retrying {retries}/{max_retries} in {delay} seconds...")
               time.sleep(delay)  # Wait before retrying
           else:
               raise
   logging.error("Failed to execute query after maximum retries due to persistent deadlocks.")
   raise RuntimeError("Query execution failed after multiple retries due to deadlock.")
           
def handle_backticks_nodes_relationship_id_type(graph_document_list:List[GraphDocument]):
  for graph_document in graph_document_list:
    # Clean node id and types
    cleaned_nodes = []
    for node in graph_document.nodes:
      if node.type.strip() and node.id.strip():
        node.type = node.type.replace('`', '')
        cleaned_nodes.append(node)
    # Clean relationship id types and source/target node id and types
    cleaned_relationships = []
    for rel in graph_document.relationships:
      if rel.type.strip() and rel.source.id.strip() and rel.source.type.strip() and rel.target.id.strip() and rel.target.type.strip():
        rel.type = rel.type.replace('`', '')
        rel.source.type = rel.source.type.replace('`', '')
        rel.target.type = rel.target.type.replace('`', '')
        cleaned_relationships.append(rel)
    graph_document.relationships = cleaned_relationships
    graph_document.nodes = cleaned_nodes
  return graph_document_list

def execute_graph_query(graph: Neo4jGraph, query, params=None, max_retries=3, delay=2):
   retries = 0
   while retries < max_retries:
       try:
           return graph.query(query, params) 
       except TransientError as e:
           if "DeadlockDetected" in str(e):
               retries += 1
               logging.info(f"Deadlock detected. Retrying {retries}/{max_retries} in {delay} seconds...")
               time.sleep(delay)  # Wait before retrying
           else:
               raise 
   logging.error("Failed to execute query after maximum retries due to persistent deadlocks.")
   raise RuntimeError("Query execution failed after multiple retries due to deadlock.")

def delete_uploaded_local_file(merged_file_path, file_name):
  file_path = Path(merged_file_path)
  if file_path.exists():
    file_path.unlink()
    logging.info(f'file {file_name} deleted successfully')
   
def close_db_connection(graph, api_name):
  if not graph._driver._closed:
      logging.info(f"closing connection for {api_name} api")
      # graph._driver.close()   

def formatted_time(current_time):
  formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S %Z')
  return str(formatted_time)

def last_url_segment(url):
  parsed_url = urlparse(url)
  path = parsed_url.path.strip("/")  # Remove leading and trailing slashes
  last_url_segment = path.split("/")[-1] if path else parsed_url.netloc.split(".")[0]
  return last_url_segment
