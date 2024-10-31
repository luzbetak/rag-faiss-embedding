#!/usr/bin/env python3

from pymongo import MongoClient
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import numpy as np
from bson.objectid import ObjectId
from faiss_store import FAISSVectorStore
from config import Config
from vectorization import VectorizationPipeline
import os
import pickle

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
            
            # Initialize vectorization pipeline
            self.vectorization_pipeline = VectorizationPipeline()
            
            # Load FAISS index if exists
            faiss_path = Config.FAISS_INDEX_PATH
            mapping_path = str(faiss_path).replace('.bin', '_mapping.pkl')
            
            if os.path.exists(str(faiss_path)):
                logger.info(f"Loading FAISS index from {faiss_path}")
                self.vector_store.load_index(str(faiss_path))
                
                # Load ID mapping if exists
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        self.vector_store.id_mapping = pickle.load(f)
                        self.vector_store.next_id = max(self.vector_store.id_mapping.keys()) + 1 if self.vector_store.id_mapping else 0
                        logger.info(f"Loaded ID mapping with {len(self.vector_store.id_mapping)} entries")
                else:
                    logger.warning(f"No ID mapping found at {mapping_path}, will rebuild mapping")
                    self.rebuild_id_mapping()
                
                vector_count = self.vector_store.index.ntotal
                logger.info(f"Loaded FAISS index with {vector_count} vectors")
            else:
                logger.warning(f"No FAISS index found at {faiss_path}")
            
            logger.info(f"Initialized Database with FAISS vector store")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def rebuild_id_mapping(self) -> bool:
        """Rebuild the FAISS to MongoDB ID mapping"""
        try:
            logger.info("Rebuilding FAISS to MongoDB ID mapping...")
            
            # Get all documents from MongoDB
            documents = list(self.collection.find({}))
            logger.info(f"Found {len(documents)} documents in MongoDB")
            
            if not documents:
                logger.warning("No documents found in MongoDB")
                return False
            
            # Generate embeddings for all documents
            contents = [doc['content'] for doc in documents]
            embeddings = self.vectorization_pipeline.generate_embeddings(contents)
            
            # Reset FAISS index
            self.vector_store = FAISSVectorStore(dimension=Config.VECTOR_DIMENSION)
            
            # Add vectors with new IDs
            mongo_ids = [str(doc['_id']) for doc in documents]
            self.vector_store.add_vectors(embeddings, mongo_ids)
            
            # Save the new index and mapping
            self.save_vector_store(str(Config.FAISS_INDEX_PATH))
            
            logger.info(f"Rebuilt ID mapping with {len(mongo_ids)} entries")
            return True
            
        except Exception as e:
            logger.exception("Error rebuilding ID mapping")
            return False

    def get_similar_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Find similar documents using FAISS"""
        try:
            logger.info(f"Searching for similar documents with top_k={top_k}")
            
            # Verify FAISS index
            vector_count = self.vector_store.index.ntotal
            if vector_count == 0:
                logger.warning("FAISS index is empty")
                return []
                
            logger.info(f"FAISS index contains {vector_count} vectors")
            
            # Search vectors in FAISS
            logger.info("Performing FAISS search...")
            distances, mongo_ids = self.vector_store.search(query_embedding, top_k)
            
            if not mongo_ids:
                logger.warning("No similar vectors found in FAISS index")
                return []
            
            logger.info(f"Found {len(mongo_ids)} similar vectors")
            
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
                        similarity_score = float(1.0 / (1.0 + distance))
                        doc["score"] = similarity_score
                        documents.append(doc)
                        logger.info(f"Found document: {doc['title']} (score: {similarity_score:.3f})")
                    else:
                        logger.warning(f"Document with ID {mongo_id} not found in MongoDB")
                except Exception as e:
                    logger.error(f"Error fetching document {mongo_id}: {e}")
                    continue
            
            if documents:
                logger.info(f"Returning {len(documents)} documents")
                # Sort by score in descending order
                documents.sort(key=lambda x: x.get("score", 0), reverse=True)
            else:
                logger.warning("No matching documents found in MongoDB")
                
            return documents
            
        except Exception as e:
            logger.exception(f"Error finding similar documents: {e}")
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
                
                # Save the updated index and mapping
                self.save_vector_store(str(Config.FAISS_INDEX_PATH))
            
        except Exception as e:
            logger.error(f"Error in batch store: {e}")
            raise

    def save_vector_store(self, filepath: str):
        """Save FAISS index and ID mapping to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            self.vector_store.save_index(filepath)
            
            # Save ID mapping
            mapping_path = filepath.replace('.bin', '_mapping.pkl')
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.vector_store.id_mapping, f)
            logger.info(f"Saved ID mapping to {mapping_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise

    def load_vector_store(self, filepath: str):
        """Load FAISS index and ID mapping from disk"""
        try:
            # Load FAISS index
            self.vector_store.load_index(filepath)
            
            # Load ID mapping
            mapping_path = filepath.replace('.bin', '_mapping.pkl')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.vector_store.id_mapping = pickle.load(f)
                    self.vector_store.next_id = max(self.vector_store.id_mapping.keys()) + 1 if self.vector_store.id_mapping else 0
                logger.info(f"Loaded ID mapping with {len(self.vector_store.id_mapping)} entries")
            else:
                logger.warning(f"No ID mapping found at {mapping_path}")
                self.rebuild_id_mapping()
            
            logger.info(f"Loaded FAISS index with {self.vector_store.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def close(self):
        """Close the database connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("Database connection closed")

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            return {
                "mongodb_documents": self.collection.count_documents({}),
                "faiss_vectors": self.vector_store.index.ntotal,
                "id_mappings": len(self.vector_store.id_mapping) if hasattr(self.vector_store, 'id_mapping') else 0,
                "vector_dimension": Config.VECTOR_DIMENSION
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

