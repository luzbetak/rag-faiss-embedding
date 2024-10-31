# database.py
from pymongo import MongoClient
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import numpy as np
from bson.objectid import ObjectId
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
            # MongoDB setup
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DATABASE_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            
            # Create MongoDB indices
            self.collection.create_index([("url", 1)], unique=True)
            self.collection.create_index([("title", 1)])
            
            # Initialize FAISS vector store
            self.vector_store = FAISSVectorStore(dimension=vector_dimension)
            
            logger.info(f"Initialized Database with FAISS vector store")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def get_similar_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Find similar documents using FAISS"""
        try:
            # Search vectors in FAISS
            distances, mongo_ids = self.vector_store.search(query_embedding, top_k)
            
            if not mongo_ids:
                logger.warning("No similar vectors found in FAISS index")
                return []
            
            # Fetch documents from MongoDB
            documents = []
            for distance, mongo_id in zip(distances, mongo_ids):
                try:
                    doc = self.collection.find_one(
                        {"_id": ObjectId(mongo_id)}, 
                        {"_id": 0, "url": 1, "title": 1, "content": 1}
                    )
                    if doc:
                        # Convert distance to similarity score (1 / (1 + distance))
                        doc["score"] = float(1.0 / (1.0 + distance))
                        documents.append(doc)
                    else:
                        logger.warning(f"Document with ID {mongo_id} not found in MongoDB")
                except Exception as e:
                    logger.error(f"Error fetching document {mongo_id}: {e}")
                    continue
            
            if documents:
                logger.info(f"Found {len(documents)} similar documents")
            else:
                logger.warning("No matching documents found in MongoDB")
                
            return documents
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

    def batch_store_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """Store documents in MongoDB and their embeddings in FAISS"""
        if not documents or not embeddings:
            logger.error("No documents or embeddings to store")
            return

        try:
            # Store documents in MongoDB and collect their IDs
            mongo_ids = []
            stored_docs = 0
            
            for doc in documents:
                try:
                    result = self.collection.insert_one({
                        "url": doc["url"],
                        "title": doc["title"],
                        "content": doc["content"]
                    })
                    mongo_ids.append(str(result.inserted_id))
                    stored_docs += 1
                except Exception as e:
                    logger.error(f"Error storing document {doc.get('url')}: {e}")
                    continue
            
            # Store embeddings in FAISS
            if mongo_ids:
                self.vector_store.add_vectors(embeddings[:len(mongo_ids)], mongo_ids)
                
            logger.info(f"Stored {stored_docs} documents with embeddings")
            
        except Exception as e:
            logger.error(f"Error in batch store: {e}")
            raise

    def init_vector_store(self, dimension: int = Config.VECTOR_DIMENSION):
        """Initialize or reinitialize the vector store"""
        self.vector_store = FAISSVectorStore(dimension=dimension)

    def save_vector_store(self, filepath: str):
        """Save FAISS index to disk"""
        self.vector_store.save_index(filepath)

    def load_vector_store(self, filepath: str):
        """Load FAISS index from disk"""
        self.vector_store.load_index(filepath)

    def close(self):
        """Close the database connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("Database connection closed")

