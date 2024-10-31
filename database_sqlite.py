
# database.py - Modified for SQLite

import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import numpy as np
from faiss_store import FAISSVectorStore
from config import Config

class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, vector_dimension: int = Config.VECTOR_DIMENSION):
        if self._initialized:
            return

        try:
            # SQLite setup
            self.connection = sqlite3.connect(Config.DATABASE_PATH)
            self.cursor = self.connection.cursor()

            # Create table if it doesn't exist
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    content TEXT,
                    vector BLOB
                )
            ''')
            self.connection.commit()
            self._initialized = True
            logger.info("SQLite database initialized.")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")

    def insert_document(self, url: str, title: str, content: str, vector: np.ndarray):
        try:
            vector_bytes = vector.tobytes()
            self.cursor.execute("INSERT INTO documents (url, title, content, vector) VALUES (?, ?, ?, ?)", 
                                (url, title, content, vector_bytes))
            self.connection.commit()
            logger.info(f"Document inserted: {title}")
        except sqlite3.IntegrityError:
            logger.warning(f"Document with URL '{url}' already exists.")
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")

    def fetch_documents(self) -> List[Dict[str, Any]]:
        try:
            self.cursor.execute("SELECT url, title, content, vector FROM documents")
            rows = self.cursor.fetchall()
            documents = []
            for row in rows:
                url, title, content, vector_bytes = row
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                documents.append({"url": url, "title": title, "content": content, "vector": vector})
            return documents
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            return []

    def close(self):
        self.connection.close()
