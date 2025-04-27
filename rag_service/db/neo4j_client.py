# rag_service/db/neo4j_client.py

from neo4j import GraphDatabase
import os

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def create_relationship(self, question: str, answer: str):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (q:Question {text: $question})
                MERGE (a:Answer {text: $answer})
                MERGE (q)-[:ANSWERED_BY]->(a)
                """,
                question=question,
                answer=answer
            )

    # It will find the data from neo4j
    def find_answer(self, question: str) -> str:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (q:Question)-[:ANSWERED_BY]->(a:Answer)
                WHERE toLower(q.text) CONTAINS toLower($question)
                RETURN a.text AS answer
                LIMIT 1
                """,
                question=question
            )
            record = result.single()
            if record:
                return record["answer"]
            return None

