# faiss_store.py

import os
import faiss
import numpy as np
from loguru import logger
from typing import Tuple, List, Optional
import pickle

class FAISSVectorStore:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FAISSVectorStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, dimension: int = 384, index_path: str = "data/faiss_index.bin"):
        """Initialize FAISS store with given dimension"""
        if self._initialized:
            return
            
        self.dimension = dimension
        self.index_path = index_path
        self.doc_ids = []  # To store mapping between FAISS and database IDs
        
        # Create L2 index
        self.index = faiss.IndexFlatL2(dimension)
        if os.path.exists(index_path):
            self.load_index()
        logger.info(f"Initialized FAISS index with dimension {dimension}")
        
        self._initialized = True

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors to the index with their corresponding IDs"""
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
            
        # Store the actual document IDs
        self.doc_ids.extend(ids)
        self.index.add(vectors)
        logger.info(f"Added {len(ids)} vectors to FAISS index with IDs: {ids}")

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[int]]:
        """Search for similar vectors in the index"""
        try:
            logger.info(f"Searching FAISS index with k={k}")
            logger.info(f"Index contains {self.index.ntotal} vectors")
            logger.debug(f"Document IDs mapping: {self.doc_ids}")  # Changed to debug level
            
            # Convert query_vector to numpy array if it's a list
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype=np.float32)
            
            # Ensure vector has correct shape
            query_vector = query_vector.reshape(1, -1)
            
            # Perform search
            distances, indices = self.index.search(query_vector, k)
            logger.info(f"Raw FAISS results - distances: {distances}, indices: {indices[0]}")
            
            # Map FAISS indices to document IDs
            doc_ids = []
            valid_distances = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.doc_ids):
                    doc_ids.append(self.doc_ids[idx])
                    valid_distances.append(distances[0][i])
                    logger.debug(f"Mapped FAISS index {idx} to document ID {self.doc_ids[idx]}")  # Changed to debug level
            
            logger.info(f"Mapped to document IDs: {doc_ids}")
            return np.array(valid_distances), doc_ids
            
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            return np.array([]), []

    def save_index(self, filepath: Optional[str] = None):
        """Save FAISS index and ID mapping to disk"""
        save_path = filepath or self.index_path
        mapping_path = save_path + '.mapping'
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, save_path)
        
        # Save ID mapping
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.doc_ids, f)
            
        logger.info(f"Saved FAISS index and mapping to {save_path}")

    def load_index(self, filepath: Optional[str] = None):
        """Load FAISS index and ID mapping from disk"""
        load_path = filepath or self.index_path
        mapping_path = load_path + '.mapping'
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(load_path)
            
            # Load ID mapping if it exists
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.doc_ids = pickle.load(f)
                logger.info(f"Loaded ID mapping for {len(self.doc_ids)} documents")
            else:
                # Create sequential IDs if no mapping file exists
                self.doc_ids = list(range(self.index.ntotal))
                logger.warning(f"No mapping file found. Created sequential IDs: {self.doc_ids}")
            
            logger.info(f"Loaded FAISS index from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise

    def reset(self):
        """Reset the index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.doc_ids = []
        logger.info("Reset FAISS index")
