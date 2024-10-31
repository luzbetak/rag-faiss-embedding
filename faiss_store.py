# faiss_store.py
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger
import os

class FAISSVectorStore:
    def __init__(self, dimension: int, index_type: str = "L2"):
        """Initialize FAISS vector store"""
        self.dimension = dimension
        self.index_type = index_type
        
        # Create base FAISS index
        if index_type == "L2":
            base_index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":
            base_index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError("index_type must be 'L2' or 'IP'")
            
        # Wrap with IDMap to support add_with_ids
        self.index = faiss.IndexIDMap(base_index)
        
        # Initialize ID mappings
        self.id_mapping = {}
        self.next_id = 0
        
        logger.info(f"Initialized FAISS index with dimension {dimension}")
    
    def add_vectors(self, vectors: List[List[float]], mongo_ids: List[str]):
        """Add vectors to the FAISS index"""
        if not vectors or not mongo_ids:
            logger.warning("No vectors or IDs provided")
            return
            
        if len(vectors) != len(mongo_ids):
            raise ValueError("Number of vectors and IDs must match")
        
        try:
            vectors_np = np.array(vectors).astype('float32')
            faiss_ids = np.arange(self.next_id, self.next_id + len(vectors), dtype='int64')
            
            # Add to FAISS index
            self.index.add_with_ids(vectors_np, faiss_ids)
            
            # Update mapping
            for faiss_id, mongo_id in zip(faiss_ids, mongo_ids):
                self.id_mapping[int(faiss_id)] = mongo_id
                
            self.next_id += len(vectors)
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            raise
    
    def search(self, query_vector: List[float], k: int = 5) -> Tuple[List[float], List[str]]:
        """Search for similar vectors"""
        try:
            query_np = np.array([query_vector]).astype('float32')
            distances, faiss_ids = self.index.search(query_np, k)
            
            # Filter valid results and convert IDs
            valid_results = []
            valid_distances = []
            
            for dist, idx in zip(distances[0], faiss_ids[0]):
                if idx != -1:  # Valid FAISS ID
                    mongo_id = self.id_mapping.get(int(idx))
                    if mongo_id:
                        valid_results.append(mongo_id)
                        valid_distances.append(float(dist))
            
            return valid_distances, valid_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [], []
    
    def save_index(self, filepath: str):
        """Save FAISS index to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            faiss.write_index(self.index, filepath)
            logger.info(f"Saved FAISS index to {filepath}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self, filepath: str):
        """Load FAISS index from disk"""
        try:
            self.index = faiss.read_index(filepath)
            logger.info(f"Loaded FAISS index from {filepath}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

