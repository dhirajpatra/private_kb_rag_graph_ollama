import logging
from langchain.docstore.document import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_models import ChatOllama
from .shared.constants import ADDITIONAL_INSTRUCTIONS
import re
import json

def get_llm(model: str):
    """Retrieve the specified language model based on the model name."""
    model = model.lower().strip()
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = os.environ.get(env_key)

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logging.error(err)
        raise Exception(err)
    
    logging.info("Model: {}".format(env_key))
    try:
        if "openai" in model:
            model_name, api_key = env_value.split(",")
            if "o3-mini" in model:
                llm= ChatOpenAI(
                api_key=api_key,
                model=model_name)
            else:
                llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0,
                )

        elif "ollama" in model:
            model_name, base_url = env_value.split(",")
            llm = ChatOllama(base_url=base_url, model=model_name)
        
        else: 
            model_name, api_endpoint, api_key = env_value.split(",")
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
            )
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)
 
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name


def get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine):
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list

def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
       Document(
           page_content=document["chunk_doc"].page_content,
           metadata={"chunk_id": [document["chunk_id"]]},
       )
       for document in chunkId_chunkDoc_list
   ]
    return combined_chunk_document_list
      

def get_graph_document_list(
    llm, combined_chunk_document_list, allowedNodes, allowedRelationship, additional_instructions=None
):
    if additional_instructions:
        additional_instructions = sanitize_additional_instruction(additional_instructions)
    graph_document_list = []
    if "get_name" in dir(llm) and llm.get_name() != "ChatOpenAI" or llm.get_name() != "AzureChatOpenAI" or llm.get_name() == "ollama":
        node_properties = False
        relationship_properties = False
    else:
        node_properties = ["description"]
        relationship_properties = ["description"]
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=node_properties,
        relationship_properties=relationship_properties,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
        ignore_tool_usage=True,
        additional_instructions=ADDITIONAL_INSTRUCTIONS+ (additional_instructions if additional_instructions else "")
    )
    
    if isinstance(llm,DiffbotGraphTransformer):
        graph_document_list = llm_transformer.convert_to_graph_documents(combined_chunk_document_list)
    else:
        graph_document_list = llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
    return graph_document_list

def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship, chunks_to_combine, additional_instructions=None):
    
    llm, model_name = get_llm(model)
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine)
    
    allowedNodes = allowedNodes.split(',') if allowedNodes else []

    if not allowedRelationship:
        allowedRelationship = []
    else:
        items = allowedRelationship.split(',')
        allowedRelationship = [tuple(items[i:i+3]) for i in range(0, len(items), 3)]
    graph_document_list = get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship, additional_instructions
    )
    return graph_document_list

def sanitize_additional_instruction(instruction: str) -> str:
   """
   Sanitizes additional instruction by:
   - Replacing curly braces `{}` with `[]` to prevent variable interpretation.
   - Removing potential injection patterns like `os.getenv()`, `eval()`, `exec()`.
   - Stripping problematic special characters.
   - Normalizing whitespace.
   Args:
       instruction (str): Raw additional instruction input.
   Returns:
       str: Sanitized instruction safe for LLM processing.
   """
   logging.info("Sanitizing additional instructions")
   instruction = instruction.replace("{", "[").replace("}", "]")  # Convert `{}` to `[]` for safety
   # Step 2: Block dangerous function calls
   injection_patterns = [r"os\.getenv\(", r"eval\(", r"exec\(", r"subprocess\.", r"import os", r"import subprocess"]
   for pattern in injection_patterns:
       instruction = re.sub(pattern, "[BLOCKED]", instruction, flags=re.IGNORECASE)
   # Step 4: Normalize spaces
   instruction = re.sub(r'\s+', ' ', instruction).strip()
   return instruction
