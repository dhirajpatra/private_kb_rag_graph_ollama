# rag_service/graph/rag_graph.py

import os
import logging
from dotenv import load_dotenv
from typing import List, TypedDict, Optional
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
    category: Optional[str]
    prompt: ChatPromptTemplate
    llm: ChatOllama

class RAGGraphService:
    def __init__(self):
        self.model = os.getenv("MODEL")
        self.base_url = os.getenv("BASE_URL")
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0.3,      # Lower randomness = faster + more deterministic
            max_tokens=256,       # Reduces generation time
            top_p=0.95            # Broader but still efficient sampling
        )

        self.prompt = self._get_prompt()
        self.category_prompt = self._get_category_prompt()

        self.neo4j_client = Neo4jClient()
        self.neo4j_client.create_indexes()

        if os.getenv("CLEAN_NEO4J_COLLECTIONS", "no").lower() == "yes":
            logging.info("Cleaning Neo4j database as per .env setting")
            self.neo4j_client.clear_database()

    def _get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
            "You are a friendly assistant for resolving user queries. "
            "Answer using the provided context. "
            "If the context does not contain the answer, reply briefly with: "
            "'I don't have that information based on the provided context.' "
            "Avoid repeating you are an offline assistant unless explicitly asked. "
            "Be brief, accurate, and stay within the bounds of the context."),
            MessagesPlaceholder(variable_name="context"),
            ("human", "{question}")
        ])

    def _get_category_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
             "Classify the following question into one of these categories: "
             "[technology, science, health, business, entertainment, general]. "
             "Respond with only the category name in lowercase."),
            ("human", "{question}")
        ])

    def _determine_category(self, question: str) -> str:
        messages = self.category_prompt.invoke({"question": question})
        response = self.llm.invoke(messages)
        return response.content.strip().lower()

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

        category = self._determine_category(state["question"])
        self.neo4j_client.create_relationship(
            question=state["question"],
            answer=answer,
            category=category
        )
        return {"answer": answer, "category": category}

    def _create_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def run(self, query: str) -> str:
        existing_answer = self.neo4j_client.find_answer(query)
        if existing_answer:
            logging.info("Answer found in Neo4j graph")
            self.neo4j_client.close()
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
        self.neo4j_client.close()
        return result["answer"]
