#!/usr/bin/env python3
import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
from loguru import logger
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch

# Configuration
DB_PATH = "data/documents.db"
FAISS_INDEX_PATH = "data/faiss_index.bin"
JSON_PATH = "data/documents.json"  # Source documents
VECTOR_DIMENSION = 384  # Set this to match the embedding dimension
TOP_K = 5  # Number of similar documents to retrieve
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Database:
    def __init__(self, db_path: str = DB_PATH):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()
        logger.info("Initialized SQLite database")

    def _create_table(self):
        """Create documents table if it doesn't exist"""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        ''')
        self.conn.commit()

    def insert_documents(self, documents: List[Dict]):
        """Insert documents into SQLite"""
        now = datetime.utcnow().isoformat()
        for doc in documents:
            try:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO documents 
                    (id, url, title, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    doc["id"],
                    doc["url"],
                    doc["title"],
                    doc["content"],
                    doc.get("created_at", now),
                    doc.get("updated_at", now)
                ))
            except sqlite3.Error as e:
                logger.error(f"Error inserting document {doc.get('url')}: {e}")
        self.conn.commit()
        logger.info(f"Inserted {len(documents)} documents into SQLite database")

    def fetch_document(self, doc_id: int) -> Optional[Dict]:
        """Fetch a document by ID"""
        self.cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "url": row[1],
                "title": row[2],
                "content": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            }
        return None

    def fetch_all_documents(self) -> List[Dict]:
        """Fetch all documents"""
        self.cursor.execute('SELECT * FROM documents')
        rows = self.cursor.fetchall()
        return [{
            "id": row[0],
            "url": row[1],
            "title": row[2],
            "content": row[3],
            "created_at": row[4],
            "updated_at": row[5]
        } for row in rows]

    def close(self):
        self.conn.close()
        logger.info("SQLite database connection closed")

class EmbeddingModel:
    def __init__(self):
        """Initialize the embedding model"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Initialized embedding model on {self.device}")

    @torch.no_grad()
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and encode text
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model output
            outputs = self.model(**inputs)
            
            # Use CLS token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

class RAGDatabaseManager:
    def __init__(self):
        self.db = Database()
        self.embedding_model = EmbeddingModel()
        self.faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        logger.info("Initialized RAG database components")

    def load_documents(self) -> List[Dict]:
        """Load documents from JSON file"""
        try:
            if not os.path.exists(JSON_PATH):
                logger.error(f"Documents file not found: {JSON_PATH}")
                return []

            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                documents = json.load(f)
                logger.info(f"Loaded {len(documents)} documents from {JSON_PATH}")
                return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    def initialize_database(self):
        """Initialize the database and FAISS index"""
        try:
            # Load documents from JSON
            documents = self.load_documents()
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
            self._save_faiss_index(documents)
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _save_faiss_index(self, documents: List[Dict]):
        """Save FAISS index and document ID mapping"""
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
            
            # Save document ID mapping
            doc_ids = [doc["id"] for doc in documents]
            mapping_path = f"{FAISS_INDEX_PATH}.mapping"
            with open(mapping_path, 'wb') as f:
                import pickle
                pickle.dump(doc_ids, f)
                
            logger.info(f"Saved FAISS index with {len(documents)} vectors")
            logger.info(f"Saved document ID mapping to {mapping_path}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    def load_indices(self):
        """Load existing FAISS index if available"""
        if os.path.exists(FAISS_INDEX_PATH):
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info("Loaded existing FAISS index")
        else:
            logger.warning("No existing FAISS index found")
            self.initialize_database()

    def search_similar_documents(self, query: str, k: int = TOP_K) -> List[Dict]:
        """Search for similar documents using FAISS"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.generate_embeddings([query])[0].reshape(1, -1)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Load document ID mapping
            with open(f"{FAISS_INDEX_PATH}.mapping", 'rb') as f:
                import pickle
                doc_ids = pickle.load(f)
            
            # Fetch documents from database
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                doc_id = doc_ids[idx]
                doc = self.db.fetch_document(doc_id)
                if doc:
                    doc['distance'] = float(distance)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def cleanup(self):
        """Cleanup resources"""
        self.db.close()

def main():
    """Main entry point for initializing the RAG database"""
    try:
        # Remove existing database and indices if they exist
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(f"{FAISS_INDEX_PATH}.mapping"):
            os.remove(f"{FAISS_INDEX_PATH}.mapping")

        # Initialize and populate the database
        manager = RAGDatabaseManager()
        manager.initialize_database()
        logger.info("Database initialization completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        if 'manager' in locals():
            manager.cleanup()

if __name__ == "__main__":
    main()
