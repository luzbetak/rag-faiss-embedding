#!/usr/bin/env python3

from pymongo import MongoClient
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import numpy as np
from bson.objectid import ObjectId
from faiss_store import FAISSVectorStore
from config import Config
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
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DATABASE_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            self.collection.create_index([("url", 1)], unique=True)
            self.collection.create_index([("title", 1)])
            
            self.vector_store = FAISSVectorStore(dimension=vector_dimension)
            faiss_path = Config.FAISS_INDEX_PATH
            mapping_path = str(faiss_path).replace('.bin', '_mapping.pkl')
            
            if os.path.exists(faiss_path):
                self.vector_store.load_index(faiss_path)
                with open(mapping_path, 'rb') as f:
                    self.vector_store.id_mapping = pickle.load(f)
                    self.vector_store.next_id = max(self.vector_store.id_mapping.keys()) + 1 if self.vector_store.id_mapping else 0
            else:
                logger.warning(f"No FAISS index found at {faiss_path}")

            self._initialized = True
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def rebuild_id_mapping(self, embeddings: List[List[float]], mongo_ids: List[str]) -> bool:
        try:
            self.vector_store.reset(dimension=Config.VECTOR_DIMENSION)
            self.vector_store.add_vectors(embeddings, mongo_ids)
            self.save_vector_store(Config.FAISS_INDEX_PATH)
            return True
        except Exception as e:
            logger.error(f"Error rebuilding ID mapping: {e}")
            return False

    def get_similar_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        distances, mongo_ids = self.vector_store.search(query_embedding, top_k)
        return [{"id": mongo_id, "score": float(1 / (1 + distance))} for distance, mongo_id in zip(distances, mongo_ids) if mongo_id]

    def batch_store_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        mongo_ids = [str(self.collection.insert_one(doc).inserted_id) for doc in documents]
        self.vector_store.add_vectors(embeddings, mongo_ids)
        self.save_vector_store(Config.FAISS_INDEX_PATH)

    def save_vector_store(self, filepath: str):
        self.vector_store.save_index(filepath)
        with open(filepath.replace('.bin', '_mapping.pkl'), 'wb') as f:
            pickle.dump(self.vector_store.id_mapping, f)

    def close(self):
        if hasattr(self, 'client'):
            self.client.close()

