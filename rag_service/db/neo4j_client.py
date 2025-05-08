# rag_service/db/neo4j_client.py

from neo4j import GraphDatabase
import neo4j
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def create_relationship(self, question: str, answer: str, category: str = None):
        """Create relationship between question and answer with optional category"""
        with self.driver.session() as session:
            query = """
                MERGE (q:Question {text: $question})
                MERGE (a:Answer {text: $answer})
                MERGE (q)-[r:ANSWERED_BY {last_used: datetime()}]->(a)
                """
            params = {"question": question, "answer": answer}
            
            if category:
                query += """
                    MERGE (c:Category {name: $category})
                    MERGE (q)-[:HAS_CATEGORY]->(c)
                    """
                params["category"] = category.lower()
            
            session.run(query, **params)

    def find_answer(self, question: str, update_last_used: bool = True) -> str:
        """Find answer for a question, optionally updating last_used timestamp"""
        with self.driver.session() as session:
            query = """
                MATCH (q:Question)-[r:ANSWERED_BY]->(a:Answer)
                WHERE toLower(q.text) CONTAINS toLower($question)
                RETURN a.text AS answer, id(r) as rel_id
                LIMIT 1
                """
            result = session.run(query, question=question)
            record = result.single()
            
            if record:
                if update_last_used:
                    # Update the last_used timestamp
                    session.run(
                        """
                        MATCH ()-[r]->()
                        WHERE id(r) = $rel_id
                        SET r.last_used = datetime()
                        """,
                        rel_id=record["rel_id"]
                    )
                return record["answer"]
            return None

    def get_questions_by_category(self, category: str) -> list:
        """Get all questions in a specific category"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (q:Question)-[:HAS_CATEGORY]->(c:Category {name: $category})
                RETURN q.text AS question
                """,
                category=category.lower()
            )
            return [record["question"] for record in result]

    def get_recently_used_answers(self, days: int = 7) -> list:
        """Get answers accessed within the last X days"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (q:Question)-[r:ANSWERED_BY]->(a:Answer)
                WHERE r.last_used >= datetime().epochMillis - ($days * 24 * 60 * 60 * 1000)
                RETURN q.text AS question, a.text AS answer, r.last_used AS last_used
                ORDER BY r.last_used DESC
                """,
                days=str(days)
            )
            return [dict(record) for record in result]
        
    def clear_database(self):
        """Delete all nodes and relationships from the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_indexes(self):
        """Ensure required indexes exist, only if the nodes are present"""
        with self.driver.session() as session:
            # Step 1: Check if any nodes of type :Question exist
            question_check = session.run("MATCH (q:Question) RETURN COUNT(q) AS count").single()
            if question_check["count"] > 0:
                try:
                    session.run("CREATE INDEX question_text_index FOR (q:Question) ON (q.text)")
                except neo4j.exceptions.Neo4jError:
                    pass  # Handle case where index already exists
            
            # Step 2: Check if any nodes of type :Category exist
            category_check = session.run("MATCH (c:Category) RETURN COUNT(c) AS count").single()
            if category_check["count"] > 0:
                try:
                    session.run("CREATE INDEX category_name_index FOR (c:Category) ON (c.name)")
                except neo4j.exceptions.Neo4jError:
                    pass  # Handle case where index already exists


    