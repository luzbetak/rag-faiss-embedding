#!/usr/bin/env python3
import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
from loguru import logger
from transformers import AutoTokenizer, AutoModel
import torch

# Configuration
DB_PATH = "data/documents.db"
FAISS_INDEX_PATH = "data/faiss_index.bin"
VECTOR_DIMENSION = 384  # Set this to match the embedding dimension
TOP_K = 5  # Number of similar documents to retrieve
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Database:
    def __init__(self):
        # Initialize SQLite3 connection
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
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

    def insert_documents(self, documents: List[Dict]):
        # Insert documents into SQLite
        for doc in documents:
            try:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO documents (url, title, content)
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

class EmbeddingModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Initialized embedding model on {self.device}")

    @torch.no_grad()
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Tokenize and encode text
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model output
            outputs = self.model(**inputs)
            
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
        
        return np.array(embeddings)

class RAGDatabaseInitializer:
    def __init__(self):
        self.db = Database()
        self.embedding_model = EmbeddingModel()
        self.faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        logger.info("Initialized RAG database components")

    def process_python_files(self, directory: str = ".") -> List[Dict]:
        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append({
                                "url": file_path,
                                "title": file,
                                "content": content
                            })
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
        return documents

    def initialize_database(self):
        # Process Python files
        documents = self.process_python_files()
        if not documents:
            logger.warning("No documents found to process")
            return

        # Insert documents into SQLite
        self.db.insert_documents(documents)
        
        # Generate embeddings
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.generate_embeddings(contents)
        
        # Add to FAISS index
        self.faiss_index.add(embeddings)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
        logger.info(f"Saved FAISS index with {len(documents)} vectors")

    def load_indices(self):
        # Load FAISS index if it exists
        if os.path.exists(FAISS_INDEX_PATH):
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"Loaded existing FAISS index")
        else:
            logger.warning("No existing FAISS index found. Initializing new database.")
            self.initialize_database()

def main():
    initializer = RAGDatabaseInitializer()
    initializer.initialize_database()
    logger.info("Database initialization completed")

if __name__ == "__main__":
    main()
