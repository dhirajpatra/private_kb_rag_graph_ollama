from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
from sse_starlette.sse import EventSourceResponse
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
from typing import List, Optional
from datetime import datetime, timezone

# Local imports
from .src.main import *
from .src.QA_integration import *
from .src.shared.common_fn import *
from .src.api_response import create_api_response
from .src.graphDB_dataAccess import graphDBdataAccess
from .src.graph_query import get_graph_results, get_chunktext_results, visualize_schema
from .src.chunkid_entities import get_entities_from_chunkids
from .src.post_processing import create_vector_fulltext_indexes, create_entity_embedding, graph_schema_consolidation
from .src.communities import create_communities
from .src.neighbours import get_neighbour_nodes
from .src.entities.source_node import sourceNode

# Standard library imports
import uvicorn
import asyncio
import base64
import logging
import time
import gc
import json
import os
load_dotenv()
logging.basicConfig(level=logging.INFO)

logger = logging
CHUNK_DIR = os.path.join(os.path.dirname(__file__), "chunks")
MERGED_DIR = os.path.join(os.path.dirname(__file__), "merged_files")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_URI_WEB = os.getenv("NEO4J_URI_WEB")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
MODEL=os.getenv("MODEL")

def sanitize_filename(filename):
   """
   Sanitize the user-provided filename to prevent directory traversal and remove unsafe characters.
   """
   # Remove path separators and collapse redundant separators
   filename = os.path.basename(filename)
   filename = os.path.normpath(filename)
   return filename

def validate_file_path(directory, filename):
   """
   Construct the full file path and ensure it is within the specified directory.
   """
   file_path = os.path.join(directory, filename)
   abs_directory = os.path.abspath(directory)
   abs_file_path = os.path.abspath(file_path)
   # Ensure the file path starts with the intended directory path
   if not abs_file_path.startswith(abs_directory):
       raise ValueError("Invalid file path")
   return abs_file_path

def healthy_condition():
    output = {"healthy": True}
    return output

def healthy():
    return True

def sick():
    return False

# Create custom middleware
class GZipPathsMiddleware:
    def __init__(self, app, paths: List[str], minimum_size: int = 1000, compresslevel: int = 5):
        self.app = app
        self.paths = paths
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel

    def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        if any(scope["path"].startswith(p) for p in self.paths):
            middleware = GZipMiddleware(
                app=self.app,
                minimum_size=self.minimum_size,
                compresslevel=self.compresslevel
            )
            return middleware(scope, receive, send)

        return self.app(scope, receive, send)

# Define FastAPI app
app = FastAPI()

# Add middlewares
app.add_middleware(
    GZipPathsMiddleware,
    paths=[
        "/sources_list", "/url/scan", "/extract", "/chat_bot",
        "/chunk_entities", "/get_neighbours", "/graph_query", "/schema",
        "/populate_graph_schema", "/get_unconnected_nodes_list",
        "/get_duplicate_nodes", "/fetch_chunktext", "/schema_visualization"
    ],
    minimum_size=1000,
    compresslevel=5
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add health route
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/url/scan")
def create_source_knowledge_graph_url(
    source_url,
    wiki_query,
    source_type,
    email: str = "anonymous",
    uri=NEO4J_URI,
    userName=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    model=MODEL,
):
    try:
        start = time.time()
        # either of one
        source = source_url or wiki_query
        graph = create_graph_database_connection(uri, userName, password, database)

        handlers = {
            'web-url': create_source_node_graph_web_url,
            'youtube': create_source_node_graph_url_youtube,
            'Wikipedia': create_source_node_graph_url_wikipedia
        }

        if source_type not in handlers:
            return create_api_response('Failed', message='Invalid source_type')

        func = handlers[source_type]
        args = (graph, model, source, source_type)
        lst_file_name, success_count, failed_count = func(*args)

        elapsed_time = time.time() - start
        message = f"Source Node created successfully for source type: {source_type} and source: {source}"
        json_obj = {
            'api_name': 'url_scan', 'db_url': uri, 'url_scanned_file': lst_file_name,
            'source_url': source_url, 'wiki_query': wiki_query, 'userName': userName,
            'database': database, 'model': model, 'source_type': source_type,
            'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time': f'{elapsed_time:.2f}',
            'email': email
        }
        logger.info(json_obj)
        return create_api_response("Success", message=message, success_count=success_count,
                                   failed_count=failed_count, file_name=lst_file_name,
                                   data={'elapsed_api_time': f'{elapsed_time:.2f}'})
    except LLMGraphBuilderException as e:
        msg = f" Unable to create source node for source type: {source_type} and source: {source}"
        error_message = str(e)
        json_obj = {
            'error_message': error_message, 'status': 'Success', 'db_url': uri,
            'userName': userName, 'database': database, 'success_count': 1,
            'source_type': source_type, 'source_url': source_url, 'wiki_query': wiki_query,
            'logging_time': formatted_time(datetime.now(timezone.utc)), 'email': email
        }
        logger.info(json_obj)
        logging.exception(f'File Failed in upload: {e}')
        return create_api_response('Failed', message=msg + error_message[:80],
                                   error=error_message, file_source=source_type)
    except Exception as e:
        msg = f" Unable to create source node for source type: {source_type} and source: {source}"
        error_message = str(e)
        json_obj = {
            'error_message': error_message, 'status': 'Failed', 'db_url': uri,
            'userName': userName, 'database': database, 'failed_count': 1,
            'source_type': source_type, 'source_url': source_url, 'wiki_query': wiki_query,
            'logging_time': formatted_time(datetime.now(timezone.utc)), 'email': email
        }
        logger.info(json_obj, "ERROR")
        logging.exception(f'Exception Stack trace upload: {e}')
        return create_api_response('Failed', message=msg + error_message[:80],
                                   error=error_message, file_source=source_type)
    finally:
        gc.collect()

@app.post("/extract")
def extract_knowledge_graph_from_file(
    source_url,
    wiki_query, 
    source_type, 
    file_name,
    allowedNodes, 
    allowedRelationship,
    token_chunk_size: Optional[int], 
    chunk_overlap: Optional[int],
    chunks_to_combine: Optional[int], 
    language,
    retry_condition, 
    additional_instructions, 
    email: str = "anonymous",
    uri=NEO4J_URI, 
    userName=NEO4J_USER, 
    password=NEO4J_PASSWORD,
    model=MODEL, 
    database=NEO4J_DATABASE
):
    try:
        start = time.time()
        file_name = sanitize_filename(file_name)
        graph = create_graph_database_connection(uri, userName, password, database)
        graph_access = graphDBdataAccess(graph)

        if source_type == 'local file':
            path = validate_file_path(MERGED_DIR, file_name)
            uri_latency, result = extract_graph_from_file_local_file(uri, userName, password, database, model, path, file_name, allowedNodes, allowedRelationship, token_chunk_size, chunk_overlap, chunks_to_combine, retry_condition, additional_instructions)
        elif source_type == 'web-url':
            uri_latency, result = extract_graph_from_web_page(uri, userName, password, database, model, source_url, file_name, allowedNodes, allowedRelationship, token_chunk_size, chunk_overlap, chunks_to_combine, retry_condition, additional_instructions)
        elif source_type == 'youtube' and source_url:
            uri_latency, result = extract_graph_from_file_youtube(uri, userName, password, database, model, source_url, file_name, allowedNodes, allowedRelationship, token_chunk_size, chunk_overlap, chunks_to_combine, retry_condition, additional_instructions)
        elif source_type == 'Wikipedia' and wiki_query:
            uri_latency, result = extract_graph_from_file_Wikipedia(uri, userName, password, database, model, wiki_query, language, file_name, allowedNodes, allowedRelationship, token_chunk_size, chunk_overlap, chunks_to_combine, retry_condition, additional_instructions)
        else:
            return create_api_response('Failed', message='Invalid source_type')

        if result:
            count = graph_access.update_node_relationship_count(file_name)
            result.update({k: count[file_name].get(k, "0") for k in [
                'chunkNodeCount', 'chunkRelCount', 'entityNodeCount', 'entityEntityRelCount',
                'communityNodeCount', 'communityRelCount', 'nodeCount', 'relationshipCount'
            ]})
            result.update({
                'db_url': uri, 'api_name': 'extract', 'source_url': source_url,
                'wiki_query': wiki_query, 'source_type': source_type,
                'logging_time': formatted_time(datetime.now(timezone.utc)),
                'elapsed_api_time': f'{time.time() - start:.2f}', 'userName': userName,
                'database': database, 'language': language, 'retry_condition': retry_condition, 'email': email
            })
            logger.info(result)
            result.update(uri_latency)
        return create_api_response('Success', data=result, file_source=source_type)

    except LLMGraphBuilderException as e:
        msg = str(e)
        graph_access = graphDBdataAccess(create_graph_database_connection(uri, userName, password, database))
        graph_access.update_exception_db(file_name, msg, retry_condition)
        if source_type == 'local file': failed_file_process(uri, file_name, path)
        node = graph_access.get_current_status_document_node(file_name)
        log_data = {
            'api_name': 'extract', 'message': msg, 'file_name': file_name,
            'file_created_at': formatted_time(node[0]['created_time']), 'status': 'Completed',
            'db_url': uri, 'userName': userName, 'database': database,
            'source_type': source_type, 'source_url': source_url, 'wiki_query': wiki_query,
            'logging_time': formatted_time(datetime.now(timezone.utc)), 'email': email
        }
        logger.info(log_data)
        logging.exception(f'Handled error: {e}')
        return create_api_response("Failed", message=msg, error=msg, file_name=file_name)

    except Exception as e:
        msg = str(e)
        graph_access = graphDBdataAccess(create_graph_database_connection(uri, userName, password, database))
        graph_access.update_exception_db(file_name, msg, retry_condition)
        if source_type == 'local file': failed_file_process(uri, file_name, path)
        node = graph_access.get_current_status_document_node(file_name)
        log_data = {
            'api_name': 'extract', 'message': f"Failed To Process File: {file_name}",
            'file_name': file_name, 'file_created_at': formatted_time(node[0]['created_time']),
            'status': 'Failed', 'error_message': msg[:300], 'db_url': uri,
            'userName': userName, 'database': database, 'source_type': source_type,
            'source_url': source_url, 'wiki_query': wiki_query,
            'logging_time': formatted_time(datetime.now(timezone.utc)), 'email': email
        }
        logger.info(log_data, "ERROR")
        logging.exception(f'Unhandled error: {e}')
        return create_api_response('Failed', message=log_data['message'] + msg[:100], error=msg, file_name=file_name)

    finally:
        gc.collect()
            
@app.post("/sources_list")
def get_source_list(
    email: str = "anonymous",
    uri=NEO4J_URI,
    userName=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
):
    """Returns list of sources from the graph database."""
    try:
        start = time.time()
        result = get_source_list_from_graph(uri, userName, password, database)
        elapsed = time.time() - start
        logger.info({
            'api_name': 'sources_list', 'db_url': uri, 'userName': userName,
            'database': database, 'logging_time': formatted_time(datetime.now(timezone.utc)),
            'elapsed_api_time': f'{elapsed:.2f}', 'email': email
        })
        return create_api_response("Success", data=result, message=f"Elapsed time: {elapsed:.2f}s")
    except Exception as e:
        logging.exception("Error fetching source list")
        return create_api_response("Failed", message="Unable to fetch source list", error=str(e))

@app.post("/post_processing")
def post_processing(
    tasks,
    email: str = "anonymous",
    uri=NEO4J_URI,
    userName=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
):
    try:
        graph = create_graph_database_connection(uri, userName, password, database)
        tasks = set(map(str.strip, json.loads(tasks)))
        api_name = 'post_processing'
        count_response = []
        start = time.time()

        if "materialize_text_chunk_similarities" in tasks:
            result = get_source_list_from_graph(uri, userName, password, database)
            logging.info("Updated KNN Graph")

        if "enable_hybrid_search_and_fulltext_search_in_bloom" in tasks:
            create_vector_fulltext_indexes(uri=uri, username=userName, password=password, database=database)
            logging.info("Full Text index created")

        if os.environ.get("ENTITY_EMBEDDING", "False").upper() == "TRUE" and "materialize_entity_similarities" in tasks:
            create_entity_embedding(graph)
            logging.info("Entity Embeddings created")

        if "graph_schema_consolidation" in tasks:
            graph_schema_consolidation,(graph)
            logging.info("Updated nodes and relationship labels")

        if "enable_communities" in tasks:
            create_communities(uri, userName, password, database)
            logging.info("Created communities")

        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb = graphDBdataAccess(graph)
        count_response = graphDb.update_node_relationship_count("")

        if count_response:
            count_response = [{"filename": f, **c} for f, c in count_response.items()]
            logging.info("Updated source node with community related counts")

        elapsed = time.time() - start
        logger.info({
            'api_name': api_name, 'db_url': uri, 'userName': userName, 'database': database,
            'logging_time': formatted_time(datetime.now(timezone.utc)),
            'elapsed_api_time': f'{elapsed:.2f}', 'email': email
        })

        return create_api_response("Success", data=count_response, message="All tasks completed successfully")

    except Exception as e:
        logging.exception("Exception in post_processing tasks")
        return create_api_response("Failed", message="Unable to complete tasks", error=str(e))

    finally:
        gc.collect()
                
@app.post("/chat_bot")
def chat_bot(
    question,
    document_names,
    session_id,
    mode,
    email: str = "anonymous",
    uri=NEO4J_URI,
    model=MODEL,
    userName=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
):
    logging.info(f"QA_RAG called at {datetime.now()}")
    qa_rag_start_time = time.time()
    
    try:
        graph = (
            Neo4jGraph(url=uri, username=userName, password=password, database=database, sanitize=True, refresh_schema=True)
            if mode == "graph"
            else create_graph_database_connection(uri, userName, password, database)
        )

        graph_access = graphDBdataAccess(graph)
        write_access = graph_access.check_account_access(database=database)

        result = QA_RAG(
            graph=graph,
            model=model,
            question=question,
            document_names=document_names,
            session_id=session_id,
            mode=mode,
            write_access=write_access
        )

        elapsed = time.time() - qa_rag_start_time
        result["info"]["response_time"] = round(elapsed, 2)

        logger.info({
            'api_name': 'chat_bot', 'db_url': uri, 'userName': userName, 'database': database,
            'question': question, 'document_names': document_names, 'session_id': session_id,
            'mode': mode, 'logging_time': formatted_time(datetime.now(timezone.utc)),
            'elapsed_api_time': f'{elapsed:.2f}', 'email': email
        })

        return create_api_response("Success", data=result)

    except Exception as e:
        logging.exception("Exception in chat bot")
        return create_api_response("Failed", message="Unable to get chat response", error=str(e), data=mode)

    finally:
        gc.collect()

@app.post("/chunk_entities")
def chunk_entities(
    entities,
    mode,
    email: str = "anonymous",
    uri=NEO4J_URI,
    userName=NEO4J_USER,
):
    try:
        start = time.time()

        result = get_entities_from_chunkids(
            nodedetails=nodedetails,
            entities=entities,
            mode=mode,
            uri=uri,
            username=userName,
            password=password,
            database=database
        )

        elapsed_time = time.time() - start

        logger.info({
            'api_name': 'chunk_entities',
            'db_url': uri,
            'userName': userName,
            'database': database,
            'nodedetails': nodedetails,
            'entities': entities,
            'mode': mode,
            'logging_time': formatted_time(datetime.now(timezone.utc)),
            'elapsed_api_time': f'{elapsed_time:.2f}',
            'email': email
        })

        return create_api_response("Success", data=result, message=f"Total elapsed API time {elapsed_time:.2f}")

    except Exception as e:
        logging.exception("Exception in chunk_entities")
        return create_api_response("Failed", message="Unable to extract entities from chunk ids", error=str(e))

    finally:
        gc.collect()

@app.post("/get_neighbours")
def get_neighbours(
    elementId,
    email: str = "anonymous",
    uri=NEO4J_URI,
    userName=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
):
    try:
        start = time.time()

        result = get_neighbour_nodes(
            uri=uri,
            username=userName,
            password=password,
            database=database,
            element_id=elementId
        )

        elapsed_time = time.time() - start

        logger.info({
            'api_name': 'get_neighbours',
            'userName': userName,
            'database': database,
            'db_url': uri,
            'logging_time': formatted_time(datetime.now(timezone.utc)),
            'elapsed_api_time': f'{elapsed_time:.2f}',
            'email': email
        })

        return create_api_response("Success", data=result, message=f"Total elapsed API time {elapsed_time:.2f}")

    except Exception as e:
        logging.exception("Exception in get_neighbours")
        return create_api_response("Failed", message="Unable to extract neighbour nodes for given element ID", error=str(e))

    finally:
        gc.collect()

@app.post("/graph_query")
def graph_query(
    document_names: str,
    email: str = "anonymous",
    uri: str=NEO4J_URI,
    database: str=NEO4J_DATABASE,
    userName: str=NEO4J_USER,
    password: str=NEO4J_PASSWORD,
):
    try:
        start = time.time()

        result = get_graph_results(
            uri=uri,
            username=userName,
            password=password,
            database=database,
            document_names=document_names
        )

        elapsed_time = time.time() - start

        logger.info({
            'api_name': 'graph_query',
            'db_url': uri,
            'userName': userName,
            'database': database,
            'document_names': document_names,
            'logging_time': formatted_time(datetime.now(timezone.utc)),
            'elapsed_api_time': f'{elapsed_time:.2f}',
            'email': email
        })

        return create_api_response("Success", data=result, message=f"Total elapsed API time {elapsed_time:.2f}")

    except Exception as e:
        logging.exception("Exception in graph_query")
        return create_api_response("Failed", message="Unable to get graph query response", error=str(e))

    finally:
        gc.collect()

@app.post("/clear_chat_bot")
def clear_chat_bot(
    session_id,
    email: str = "anonymous",
    uri=NEO4J_URI,
    userName=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
):
    try:
        start = time.time()

        graph = create_graph_database_connection(uri, userName, password, database)
        result = clear_chat_history(graph=graph, session_id=session_id)

        elapsed_time = time.time() - start

        logger.info({
            'api_name': 'clear_chat_bot',
            'db_url': uri,
            'userName': userName,
            'database': database,
            'session_id': session_id,
            'logging_time': formatted_time(datetime.now(timezone.utc)),
            'elapsed_api_time': f'{elapsed_time:.2f}',
            'email': email
        })

        return create_api_response("Success", data=result)

    except Exception as e:
        logging.exception("Exception in clear_chat_bot")
        return create_api_response("Failed", message="Unable to clear chat History", error=str(e))

    finally:
        gc.collect()
            
@app.post("/connect")
def connect(email: str = "anonymous", uri=NEO4J_URI, userName=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        result = connection_check_and_get_vector_dimensions(graph, database)
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'connect','db_url':uri, 'userName':userName, 'database':database, 'count':1, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        result['elapsed_api_time'] = f'{elapsed_time:.2f}'
        return create_api_response('Success',data=result)
    except Exception as e:
        job_status = "Failed"
        message="Connection failed to connect Neo4j database"
        error_message = str(e)
        logging.exception(f'Connection failed to connect Neo4j database:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)

@app.post("/upload")
def upload_large_file_into_chunks(chunkNumber, 
                                    totalChunks, 
                                    originalname, 
                                    email: str = "anonymous",
                                    model=MODEL, 
                                    file:UploadFile = File(...),
                                    uri=NEO4J_URI, 
                                    userName=NEO4J_USER, 
                                    password=NEO4J_PASSWORD, 
                                    database=NEO4J_DATABASE,
                                    ):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        result = upload_file(graph, model, file, chunkNumber, totalChunks, originalname, uri, CHUNK_DIR, MERGED_DIR)
        end = time.time()
        elapsed_time = end - start
        if int(chunkNumber) == int(totalChunks):
            json_obj = {'api_name':'upload','db_url':uri,'userName':userName, 'database':database, 'chunkNumber':chunkNumber,'totalChunks':totalChunks,
                                'original_file_name':originalname,'model':model, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
            logger.info(json_obj)
        if int(chunkNumber) == int(totalChunks):
            return create_api_response('Success',data=result, message='Source Node Created Successfully')
        else:
            return create_api_response('Success', message=result)
    except Exception as e:
        message="Unable to upload file in chunks"
        error_message = str(e)
        graph = create_graph_database_connection(uri, userName, password, database)   
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.update_exception_db(originalname,error_message)
        logging.info(message)
        logging.exception(f'Exception:{error_message}')
        return create_api_response('Failed', message=message + error_message[:100], error=error_message, file_name = originalname)
    finally:
        gc.collect()
            
@app.post("/schema")
def get_structured_schema(email: str = "anonymous", uri=NEO4J_URI, userName=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE):
    try:
        start = time.time()
        result = get_labels_and_relationtypes(uri, userName, password, database)
        end = time.time()
        elapsed_time = end - start
        logging.info(f'Schema result from DB: {result}')
        json_obj = {'api_name':'schema','db_url':uri, 'userName':userName, 'database':database, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success', data=result,message=f"Total elapsed API time {elapsed_time:.2f}")
    except Exception as e:
        message="Unable to get the labels and relationtypes from neo4j database"
        error_message = str(e)
        logging.info(message)
        logging.exception(f'Exception:{error_message}')
        return create_api_response("Failed", message=message, error=error_message)
    finally:
        gc.collect()
            
def decode_password(pwd):
    sample_string_bytes = base64.b64decode(pwd)
    decoded_password = sample_string_bytes.decode("utf-8")
    return decoded_password

def encode_password(pwd):
    data_bytes = pwd.encode('ascii')
    encoded_pwd_bytes = base64.b64encode(data_bytes)
    return encoded_pwd_bytes

@app.get("/update_extract_status/{file_name}")
def update_extract_status(request: Request, 
                          file_name: str, 
                          uri:str=NEO4J_URI, 
                          userName:str=NEO4J_USER, 
                          password:str=NEO4J_PASSWORD, 
                          database:str=NEO4J_DATABASE):
    def generate():
        status = ''
        
        if password is not None and password != "null":
            decoded_password = decode_password(password)
        else:
            decoded_password = None

        url = uri
        if url and " " in url:
            url= url.replace(" ","+")
            
        graph = create_graph_database_connection(url, userName, decoded_password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        while True:
            try:
                if request.is_disconnected():
                    logging.info(" SSE Client disconnected")
                    break
                # get the current status of document node
                
                else:
                    result = graphDb_data_Access.get_current_status_document_node(file_name)
                    if len(result) > 0:
                        status = json.dumps({'fileName':file_name, 
                        'status':result[0]['Status'],
                        'processingTime':result[0]['processingTime'],
                        'nodeCount':result[0]['nodeCount'],
                        'relationshipCount':result[0]['relationshipCount'],
                        'model':result[0]['model'],
                        'total_chunks':result[0]['total_chunks'],
                        'fileSize':result[0]['fileSize'],
                        'processed_chunk':result[0]['processed_chunk'],
                        'fileSource':result[0]['fileSource'],
                        'chunkNodeCount' : result[0]['chunkNodeCount'],
                        'chunkRelCount' : result[0]['chunkRelCount'],
                        'entityNodeCount' : result[0]['entityNodeCount'],
                        'entityEntityRelCount' : result[0]['entityEntityRelCount'],
                        'communityNodeCount' : result[0]['communityNodeCount'],
                        'communityRelCount' : result[0]['communityRelCount']
                        })
                    yield status
            except asyncio.CancelledError:
                logging.info("SSE Connection cancelled")
    
    return EventSourceResponse(generate(),ping=60)

@app.post("/delete_document_and_entities")
def delete_document_and_entities(filenames,
                                       source_types,
                                       deleteEntities,
                                       email: str = "anonymous",
                                       uri=NEO4J_URI, 
                                       userName=NEO4J_USER, 
                                       password=NEO4J_PASSWORD, 
                                       database=NEO4J_DATABASE, 
                                       ):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        files_list_size = graphDb_data_Access.delete_file_from_graph(filenames, source_types, deleteEntities, MERGED_DIR, uri)
        message = f"Deleted {files_list_size} documents with entities from database"
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'delete_document_and_entities','db_url':uri, 'userName':userName, 'database':database, 'filenames':filenames,'deleteEntities':deleteEntities,
                            'source_types':source_types, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success',message=message)
    except Exception as e:
        job_status = "Failed"
        message=f"Unable to delete document {filenames}"
        error_message = str(e)
        logging.exception(f'{message}:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()

@app.get('/document_status/{file_name}')
def get_document_status(file_name, 
                        url, 
                        userName=NEO4J_USER, 
                          password=NEO4J_PASSWORD, 
                          database=NEO4J_DATABASE):
    decoded_password = decode_password(password)
   
    try:
        if " " in url:
            uri= url.replace(" ","+")
        else:
            uri=url
        graph = create_graph_database_connection(uri, userName, decoded_password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        result = graphDb_data_Access.get_current_status_document_node(file_name)
        if len(result) > 0:
            status = {'fileName':file_name, 
                'status':result[0]['Status'],
                'processingTime':result[0]['processingTime'],
                'nodeCount':result[0]['nodeCount'],
                'relationshipCount':result[0]['relationshipCount'],
                'model':result[0]['model'],
                'total_chunks':result[0]['total_chunks'],
                'fileSize':result[0]['fileSize'],
                'processed_chunk':result[0]['processed_chunk'],
                'fileSource':result[0]['fileSource'],
                'chunkNodeCount' : result[0]['chunkNodeCount'],
                'chunkRelCount' : result[0]['chunkRelCount'],
                'entityNodeCount' : result[0]['entityNodeCount'],
                'entityEntityRelCount' : result[0]['entityEntityRelCount'],
                'communityNodeCount' : result[0]['communityNodeCount'],
                'communityRelCount' : result[0]['communityRelCount']
                }
        else:
            status = {'fileName':file_name, 'status':'Failed'}
        logging.info(f'Result of document status in refresh : {result}')
        return create_api_response('Success',message="",file_name=status)
    except Exception as e:
        message=f"Unable to get the document status"
        error_message = str(e)
        logging.exception(f'{message}:{error_message}')
        return create_api_response('Failed',message=message)
    
@app.post("/cancelled_job")
def cancelled_job(uri, 
                  filenames, 
                  source_types,
                  userName=NEO4J_USER, 
                          password=NEO4J_PASSWORD, 
                          database=NEO4J_DATABASE,
                          email:str = "anonymous"):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        result = manually_cancelled_job(graph,filenames, source_types, MERGED_DIR, uri)
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'cancelled_job','db_url':uri, 'userName':userName, 'database':database, 'filenames':filenames,
                            'source_types':source_types, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success',message=result)
    except Exception as e:
        job_status = "Failed"
        message="Unable to cancelled the running job"
        error_message = str(e)
        logging.exception(f'Exception in cancelling the running job:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()

@app.post("/populate_graph_schema")
def populate_graph_schema(input_text, 
                          is_schema_description_checked,
                          is_local_storage,
                          model=MODEL, 
                          email: str = "anonymous"):
    try:
        start = time.time()
        result = populate_graph_schema_from_text(input_text, model, is_schema_description_checked, is_local_storage)
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'populate_graph_schema', 'model':model, 'is_schema_description_checked':is_schema_description_checked, 'input_text':input_text, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success',data=result)
    except Exception as e:
        job_status = "Failed"
        message="Unable to get the schema from text"
        error_message = str(e)
        logging.exception(f'Exception in getting the schema from text:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()
        
@app.post("/get_unconnected_nodes_list")
def get_unconnected_nodes_list(uri=NEO4J_URI, 
                               userName=NEO4J_USER, 
                          password=NEO4J_PASSWORD, 
                          database=NEO4J_DATABASE,
                          email:str = "anonymous"):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        nodes_list, total_nodes = graphDb_data_Access.list_unconnected_nodes()
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'get_unconnected_nodes_list','db_url':uri, 'userName':userName, 'database':database, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success',data=nodes_list,message=total_nodes)
    except Exception as e:
        job_status = "Failed"
        message="Unable to get the list of unconnected nodes"
        error_message = str(e)
        logging.exception(f'Exception in getting list of unconnected nodes:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()
        
@app.post("/delete_unconnected_nodes")
def delete_orphan_nodes(unconnected_entities_list,
                        uri=NEO4J_URI, 
                        userName=NEO4J_USER, 
                          password=NEO4J_PASSWORD, 
                          database=NEO4J_DATABASE,
                          email:str = "anonymous"):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        result = graphDb_data_Access.delete_unconnected_nodes(unconnected_entities_list)
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'delete_unconnected_nodes','db_url':uri, 'userName':userName, 'database':database,'unconnected_entities_list':unconnected_entities_list, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success',data=result,message="Unconnected entities delete successfully")
    except Exception as e:
        job_status = "Failed"
        message="Unable to delete the unconnected nodes"
        error_message = str(e)
        logging.exception(f'Exception in delete the unconnected nodes:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()
        
@app.post("/get_duplicate_nodes")
def get_duplicate_nodes(uri=NEO4J_URI, 
                        userName=NEO4J_USER, 
                          password=NEO4J_PASSWORD, 
                          database=NEO4J_DATABASE,
                          email:str = "anonymous"):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        nodes_list, total_nodes = graphDb_data_Access.get_duplicate_nodes_list()
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'get_duplicate_nodes','db_url':uri,'userName':userName, 'database':database, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success',data=nodes_list, message=total_nodes)
    except Exception as e:
        job_status = "Failed"
        message="Unable to get the list of duplicate nodes"
        error_message = str(e)
        logging.exception(f'Exception in getting list of duplicate nodes:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()
        
@app.post("/merge_duplicate_nodes")
def merge_duplicate_nodes(duplicate_nodes_list,
                          uri=NEO4J_URI, 
                          userName=NEO4J_USER, 
                          password=NEO4J_PASSWORD, 
                          database=NEO4J_DATABASE,
                          email:str = "anonymous"):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        result = graphDb_data_Access.merge_duplicate_nodes(duplicate_nodes_list)
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'merge_duplicate_nodes','db_url':uri, 'userName':userName, 'database':database,
                            'duplicate_nodes_list':duplicate_nodes_list, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success',data=result,message="Duplicate entities merged successfully")
    except Exception as e:
        job_status = "Failed"
        message="Unable to merge the duplicate nodes"
        error_message = str(e)
        logging.exception(f'Exception in merge the duplicate nodes:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()
        
@app.post("/drop_create_vector_index")
def drop_create_vector_index(isVectorIndexExist,
                             uri=NEO4J_URI, 
                             userName=NEO4J_USER, 
                             password=NEO4J_PASSWORD, 
                             database=NEO4J_DATABASE, 
                             email: str = "anonymous"):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        graphDb_data_Access = graphDBdataAccess(graph)
        result = graphDb_data_Access.drop_create_vector_index(isVectorIndexExist)
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'drop_create_vector_index', 'db_url':uri, 'userName':userName, 'database':database,
                            'isVectorIndexExist':isVectorIndexExist, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        return create_api_response('Success',message=result)
    except Exception as e:
        job_status = "Failed"
        message="Unable to drop and re-create vector index with correct dimesion as per application configuration"
        error_message = str(e)
        logging.exception(f'Exception into drop and re-create vector index with correct dimesion as per application configuration:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()
        
@app.post("/retry_processing")
def retry_processing(file_name, 
                     retry_condition, 
                     uri=NEO4J_URI, 
                     userName=NEO4J_USER, 
                     password=NEO4J_PASSWORD, 
                     database=NEO4J_DATABASE, 
                     email: str = "anonymous"):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        chunks = execute_graph_query(graph,QUERY_TO_GET_CHUNKS,params={"filename":file_name})
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'retry_processing', 'db_url':uri, 'userName':userName, 'database':database, 'file_name':file_name,'retry_condition':retry_condition,
                            'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.info(json_obj)
        if chunks[0]['text'] is None or chunks[0]['text']=="" or not chunks :
            return create_api_response('Success',message=f"Chunks are not created for the file{file_name}. Please upload again the file to re-process.",data=chunks)
        else:
            set_status_retry(graph, file_name, retry_condition)
            return create_api_response('Success',message=f"Status set to Ready to Reprocess for filename : {file_name}")
    except Exception as e:
        job_status = "Failed"
        message="Unable to set status to Retry"
        error_message = str(e)
        logging.exception(f'{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()    

@app.post("/fetch_chunktext")
def fetch_chunktext(
    document_name: str,
   page_no: int,
   email: str = "anonymous",
   uri: str = NEO4J_URI,
   database: str = NEO4J_DATABASE,
   userName: str = NEO4J_USER,
   password: str = NEO4J_PASSWORD,
):
   try:
       start = time.time()
       result = get_chunktext_results(
           uri=uri,
           username=userName,
           password=password,
           database=database,
           document_name=document_name,
           page_no=page_no
       )
       end = time.time()
       elapsed_time = end - start
       json_obj = {
           'api_name': 'fetch_chunktext',
           'db_url': uri,
           'userName': userName,
           'database': database,
           'document_name': document_name,
           'page_no': page_no,
           'logging_time': formatted_time(datetime.now(timezone.utc)),
           'elapsed_api_time': f'{elapsed_time:.2f}',
           'email': email
       }
       logger.info(json_obj)
       return create_api_response('Success', data=result, message=f"Total elapsed API time {elapsed_time:.2f}")
   except Exception as e:
       job_status = "Failed"
       message = "Unable to get chunk text response"
       error_message = str(e)
       logging.exception(f'Exception in fetch_chunktext: {error_message}')
       return create_api_response(job_status, message=message, error=error_message)
   finally:
       gc.collect()

@app.post("/backend_connection_configuration")
def backend_connection_configuration():
    """
    Checks the backend Neo4j graph database connection and returns its status along with vector index metadata.
    """
    try:
        start = time.time()
        
        # Load connection values from environment/config
        uri = NEO4J_URI
        username = NEO4J_USER
        database = NEO4J_DATABASE
        password = NEO4J_PASSWORD

        # Validate connection parameters
        if all([uri, username, database, password]):
            graph = Neo4jGraph(
                url=uri,
                username=username,
                password=password,
                database=database,
                refresh_schema=False,
                sanitize=True
            )

            logging.info(f'Graph object initialized: {graph}')

            if graph:
                graph_connection = True
                graphDb_data_Access = graphDBdataAccess(graph)
                result = graphDb_data_Access.connection_check_and_get_vector_dimensions(database)

                result.update({
                    'uri': uri,
                    'api_name': 'backend_connection_configuration',
                    'elapsed_api_time': f"{time.time() - start:.2f}",
                    'graph_connection': graph_connection,
                    'connection_from': 'backendAPI'
                })

                logger.info(result)
                return create_api_response('Success', message="Backend connection successful", data=result)

        # Incomplete config case
        graph_connection = False
        return create_api_response('Success', message="Backend connection not successful", data=graph_connection)

    except Exception as e:
        graph_connection = False
        error_message = str(e)
        logging.exception(error_message)
        return create_api_response(
            "Failed",
            message="Unable to connect to knowledge graph DB",
            error=error_message.rstrip('.') + ', or fill from the login dialog.',
            data=graph_connection
        )

    finally:
        gc.collect()
    
@app.post("/schema_visualization")
def get_schema_visualization(uri=NEO4J_URI, userName=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE):
    try:
        start = time.time()
        result = visualize_schema(
           uri=uri,
           userName=userName,
           password=password,
           database=database)
        if result:
            logging.info("Graph schema visualization query successful")
        end = time.time()
        elapsed_time = end - start
        logging.info(f'Schema result from DB: {result}')
        json_obj = {'api_name':'schema_visualization','db_url':uri, 'userName':userName, 'database':database, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}'}
        logger.info(json_obj)
        return create_api_response('Success', data=result,message=f"Total elapsed API time {elapsed_time:.2f}")
    except Exception as e:
        message="Unable to get schema visualization from neo4j database"
        error_message = str(e)
        logging.info(message)
        logging.exception(f'Exception:{error_message}')
        return create_api_response("Failed", message=message, error=error_message)
    finally:
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)