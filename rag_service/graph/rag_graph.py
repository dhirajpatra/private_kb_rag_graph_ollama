# rag_service/graph/rag_graph.py

import os
import logging
from dotenv import load_dotenv
from typing import List, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from db.neo4j_client import Neo4jClient
from tools.retriever_tool import retriever_tool

logging.basicConfig(level=logging.INFO)
load_dotenv()

class GraphState(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: str
    prompt: ChatPromptTemplate
    llm: ChatOllama

class RAGGraphService:
    def __init__(self):
        self.model = os.getenv("MODEL")
        self.base_url = os.getenv("BASE_URL")
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0,
            max_tokens=300,
            top_p=0.1
        )
        self.prompt = self._get_prompt()

    def _get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", 
            "You are a **private, offline assistant**. "
            "You do not have access to external internet or any personal data outside the provided context. "
            "Always answer based only on the provided context. "
            "If the answer is not available in the context, politely say: "
            "'I am a private offline assistant and can only answer based on available information.' "
            "Never create or assume any personal information. "
            "Stay concise, helpful, and respectful."),
            MessagesPlaceholder(variable_name="context"),
            ("human", "{question}")
        ])

    def _retrieve(self, state: GraphState) -> dict:
        logging.info("Entering retrieve node")
        query = state["question"]
        response = retriever_tool.invoke({"query": query, "k": 3})
        if response.get("status") != "success":
            raise ValueError("Retriever tool failed")
        docs = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in response["results"]]
        logging.info(f"Retrieved {len(docs)} documents")
        return {"context": docs}

    def _generate(self, state: GraphState) -> dict:
        logging.info("Entering generate node")
        llm = state["llm"]
        prompt = state["prompt"]
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        messages = prompt.invoke({
            "question": state["question"],
            "context": [HumanMessage(content=docs_content)]
        })

        response = llm.invoke(messages)
        answer = response.content
        logging.info(f"Generated response: {answer}")

        neo4j_client = Neo4jClient()
        neo4j_client.create_relationship(state["question"], answer)
        neo4j_client.close()

        return {"answer": answer}

    def _create_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def run(self, query: str) -> str:
        neo4j_client = Neo4jClient()
        existing_answer = neo4j_client.find_answer(query)
        neo4j_client.close()

        if existing_answer:
            logging.info("Answer found in Neo4j graph")
            return existing_answer

        rag_graph = self._create_graph()
        initial_state = GraphState(
            question=query,
            context=[],
            answer="",
            prompt=self.prompt,
            llm=self.llm
        )
        result = rag_graph.invoke(initial_state)
        return result["answer"]
