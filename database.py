# database.py

import sqlite3
from pathlib import Path
from loguru import logger
from typing import List, Dict, Optional
import numpy as np
from faiss_store import FAISSVectorStore

class Database:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "data/documents.db"):
        """Initialize SQLite database connection"""
        if self._initialized:
            return
            
        self.db_path = db_path
        Path(db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()
        
        # Initialize FAISS vector store (will reuse existing instance if any)
        self.vector_store = FAISSVectorStore()
        logger.info("Initialized Database with FAISS vector store")
        
        self._initialized = True

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
        try:
            self.cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
            row = self.cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "url": row[1],
                    "title": row[2],
                    "content": row[3]
                }
        except sqlite3.Error as e:
            logger.error(f"Error fetching document {doc_id}: {e}")
        return None

    def get_document_count(self) -> int:
        """Get total number of documents"""
        self.cursor.execute('SELECT COUNT(*) FROM documents')
        return self.cursor.fetchone()[0]

    def load_vector_store(self, filepath: str):
        """Load FAISS index from disk"""
        try:
            self.vector_store.load_index(filepath)
            logger.info(f"Loaded vector store from {filepath}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def save_vector_store(self, filepath: str):
        """Save FAISS index to disk"""
        try:
            self.vector_store.save_index(filepath)
            logger.info(f"Saved vector store to {filepath}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise

    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Database connection closed")
