# agent_service/graph/rag_graph.py
import os
import logging
from dotenv import load_dotenv
from typing_extensions import List, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from tools.retriever_tool import retriever_tool
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
load_dotenv()

# 1. Define the State
class GraphState(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: str
    prompt: ChatPromptTemplate
    llm: ChatOllama

# 2. Define the Prompt
def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the question."),
        MessagesPlaceholder(variable_name="context"),
        ("human", "{question}")
    ])

# 3. Graph Nodes
def retrieve(state: GraphState) -> dict:
    logging.info("Entering retrieve node")
    query = state["question"]
    response = retriever_tool.invoke({"query": query, "k": 3})
    if response["status"] != "success":
        raise ValueError("Retriever tool failed")
    retrieved_docs = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in response["results"]]
    logging.info(f"Retrieved {len(retrieved_docs)} documents")
    return {"context": retrieved_docs}

def generate(state: GraphState) -> dict:
    logging.info("Entering generate node")
    llm = state.get("llm")  # type: ChatOllama
    prompt = state.get("prompt")  # type: ChatPromptTemplate

    # Use the retrieved documents' content
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    # Construct the prompt using both the query and the retrieved documents
    messages = prompt.invoke({
        "question": state["question"],
        "context": [HumanMessage(content=docs_content)]  # Include documents in context
    })

    # Generate the final response
    response = llm.invoke(messages)
    logging.info(f"Generated response: {response.content}")
    
    return {"answer": response.content}

# 4. Create Graph
def create_graph(llm: ChatOllama, prompt: ChatPromptTemplate):
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()

# 5. Run RAG Chain
def run_rag_chain(query: str):
    llm = ChatOllama(
        model=os.getenv("MODEL"),
        base_url=os.getenv("BASE_URL"),
        temperature=0,
        max_tokens=500,
        top_p=0.1
    )
    prompt = get_prompt()
    rag_graph = create_graph(llm, prompt)

    initial_state = GraphState(
        question=query,
        context=[],
        answer="",
        prompt=prompt,
        llm=llm
    )

    result = rag_graph.invoke(initial_state)
    return result["answer"]

