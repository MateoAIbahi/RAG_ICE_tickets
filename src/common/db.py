"""Connexion unique à la base RAG (avant : 6 copies de get_rag_conn dans le code)."""
import os

import psycopg2

DEFAULT_URL = "postgresql://rag:ragpass@db:5432/ragdb"


def get_conn(autocommit: bool = True):
    conn = psycopg2.connect(os.getenv("RAG_DATABASE_URL", DEFAULT_URL))
    conn.autocommit = autocommit
    return conn


def vector_literal(vec) -> str:
    """Format attendu par pgvector : '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
