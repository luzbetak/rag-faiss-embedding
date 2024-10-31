#!/usr/bin/env python3
import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from loguru import logger

# Configuration
DB_PATH = "data/documents.db"
JSON_PATH = "data/search-index.json"
FAISS_INDEX_PATH = "data/faiss_index.bin"
VECTOR_DIMENSION = 384  # Set this to match the embedding dimension
TOP_K = 5  # Number of similar documents to retrieve

class Database:
    def __init__(self):
        # Initialize SQLite3 connection
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self._create_table()
        logger.info("Initialized SQLite3 database")

    def _create_table(self):
        # Create the documents table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT
        )
        ''')
        self.conn.commit()
        logger.info("Documents table created (if not already existing)")

    def insert_documents(self, documents: List[Dict]):
        # Insert documents into SQLite
        for doc in documents:
            try:
                self.cursor.execute('''
                    INSERT OR IGNORE INTO documents (url, title, content)
                    VALUES (?, ?, ?)
                ''', (doc["url"], doc["title"], doc["content"]))
            except sqlite3.Error as e:
                logger.error(f"Error inserting document {doc['url']}: {e}")
        self.conn.commit()
        logger.info(f"Inserted {len(documents)} documents into SQLite database")

    def fetch_document(self, doc_id: int) -> Optional[Dict]:
        # Fetch a document by ID
        self.cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = self.cursor.fetchone()
        return {"id": row[0], "url": row[1], "title": row[2], "content": row[3]} if row else None

    def fetch_all_documents(self) -> List[Dict]:
        # Fetch all documents
        self.cursor.execute('SELECT * FROM documents')
        rows = self.cursor.fetchall()
        return [{"id": row[0], "url": row[1], "title": row[2], "content": row[3]} for row in rows]

    def close(self):
        self.conn.close()
        logger.info("SQLite3 connection closed")

class FAISSIndex:
    def __init__(self, dimension: int, index_path: str = FAISS_INDEX_PATH):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.index_path = index_path
        self.doc_ids = []

        # Load existing index if it exists
        if os.path.exists(index_path):
            self.load_index()

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        # Use add_with_ids to associate vectors with their IDs
        faiss_ids = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, faiss_ids)
        self.doc_ids.extend(ids)
        logger.info(f"Added {len(ids)} vectors to FAISS index")

    def search(self, query_vector: np.ndarray, top_k: int = TOP_K) -> List[Tuple[int, float]]:
        # Perform a search on FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        return [(self.doc_ids[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx != -1]

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        logger.info(f"FAISS index saved at {self.index_path}")

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        logger.info(f"FAISS index loaded from {self.index_path}")

def load_documents(json_path: str) -> List[Dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} documents from {json_path}")
    return data

def generate_embeddings(documents: List[Dict]) -> np.ndarray:
    # Placeholder for embedding generation
    # Replace with actual embedding generation (e.g., using a model)
    return np.random.rand(len(documents), VECTOR_DIMENSION).astype('float32')

def main():
    # Initialize database and FAISS index
    db = Database()
    faiss_index = FAISSIndex(dimension=VECTOR_DIMENSION)

    # Load documents from JSON and insert into SQLite
    documents = load_documents(JSON_PATH)
    db.insert_documents(documents)

    # Generate embeddings and add to FAISS index
    embeddings = generate_embeddings(documents)
    doc_ids = [doc['id'] for doc in db.fetch_all_documents()]
    faiss_index.add_vectors(embeddings, doc_ids)

    # Save FAISS index to disk
    faiss_index.save_index()

    # Example search
    query_embedding = np.random.rand(1, VECTOR_DIMENSION).astype('float32')
    results = faiss_index.search(query_embedding)
    for doc_id, distance in results:
        doc = db.fetch_document(doc_id)
        if doc:
            print(f"Found document: {doc['title']} (Distance: {distance})")

    # Close database connection
    db.close()

if __name__ == "__main__":
    main()

