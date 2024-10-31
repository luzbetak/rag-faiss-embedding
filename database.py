# database.py

import sqlite3
from pathlib import Path
from loguru import logger
from typing import List, Dict, Optional
import numpy as np
from faiss_store import FAISSVectorStore

class Database:
    def __init__(self, db_path: str = "data/documents.db"):
        """Initialize SQLite database connection"""
        self.db_path = db_path
        Path(db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()
        
        # Initialize FAISS vector store
        self.vector_store = FAISSVectorStore()
        logger.info("Initialized Database with FAISS vector store")

    def _create_table(self):
        """Create documents table if it doesn't exist"""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT
        )
        ''')
        self.conn.commit()

    def insert_documents(self, documents: List[Dict]):
        """Insert documents into SQLite database"""
        for doc in documents:
            try:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO documents (url, title, content)
                    VALUES (?, ?, ?)
                ''', (doc["url"], doc["title"], doc["content"]))
            except sqlite3.Error as e:
                logger.error(f"Error inserting document {doc.get('url')}: {e}")
        self.conn.commit()
        logger.info(f"Inserted {len(documents)} documents")

    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Fetch a document by its ID"""
        self.cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "url": row[1],
                "title": row[2],
                "content": row[3]
            }
        return None

    def get_document_count(self) -> int:
        """Get total number of documents"""
        self.cursor.execute('SELECT COUNT(*) FROM documents')
        return self.cursor.fetchone()[0]

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")

