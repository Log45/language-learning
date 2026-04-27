import os
try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - fallback for environments without neo4j installed
    GraphDatabase = None


NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


driver = None
if GraphDatabase is not None:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

