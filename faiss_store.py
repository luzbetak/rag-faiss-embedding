# faiss_store.py
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger

class FAISSVectorStore:
    def __init__(self, dimension: int, index_type: str = "L2"):
        """Initialize FAISS vector store"""
        self.dimension = dimension
        self.index_type = index_type
        
        logger.debug(f"Initializing FAISS index - dimension: {dimension}, type: {index_type}")
        
        # Create base FAISS index
        if index_type == "L2":
            base_index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":
            base_index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError("index_type must be 'L2' or 'IP'")
            
        # Wrap with IDMap
        self.index = faiss.IndexIDMap(base_index)
        self.id_mapping = {}
        self.next_id = 0
        
        logger.info(f"FAISS index initialized - type: {index_type}")

    def add_vectors(self, vectors: List[List[float]], mongo_ids: List[str]):
        """Add vectors to FAISS index"""
        try:
            logger.debug(f"Adding vectors - count: {len(vectors)}, first mongo_id: {mongo_ids[0]}")
            vectors_np = np.array(vectors).astype('float32')
            faiss_ids = np.arange(self.next_id, self.next_id + len(vectors), dtype='int64')
            
            # Debug info before adding
            logger.debug(f"Vector shape: {vectors_np.shape}")
            logger.debug(f"FAISS IDs: {faiss_ids}")
            
            self.index.add_with_ids(vectors_np, faiss_ids)
            
            # Update mapping
            for faiss_id, mongo_id in zip(faiss_ids, mongo_ids):
                self.id_mapping[int(faiss_id)] = mongo_id
                logger.debug(f"Mapped FAISS ID {faiss_id} to MongoDB ID {mongo_id}")
                
            self.next_id += len(vectors)
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            logger.debug(f"Current index total: {self.index.ntotal}")
            
        except Exception as e:
            logger.exception(f"Error adding vectors: {e}")
            raise

    def search(self, query_vector: List[float], k: int = 5) -> Tuple[List[float], List[str]]:
        """Search for similar vectors"""
        try:
            logger.debug(f"Starting FAISS search with k={k}")
            logger.debug(f"Index total: {self.index.ntotal}")
            logger.debug(f"Query vector shape: {len(query_vector)}")
            
            # Print first few components of query vector
            logger.debug(f"Query vector preview: {query_vector[:5]}...")
            
            query_np = np.array([query_vector]).astype('float32')
            distances, faiss_ids = self.index.search(query_np, k)
            
            logger.debug(f"Raw search results:")
            logger.debug(f"Distances: {distances[0]}")
            logger.debug(f"FAISS IDs: {faiss_ids[0]}")
            
            # Convert IDs and filter results
            valid_results = []
            valid_distances = []
            
            for dist, idx in zip(distances[0], faiss_ids[0]):
                if idx != -1:
                    mongo_id = self.id_mapping.get(int(idx))
                    if mongo_id:
                        valid_results.append(mongo_id)
                        valid_distances.append(float(dist))
                        logger.debug(f"Valid result - distance: {dist:.3f}, "
                                   f"FAISS ID: {idx}, MongoDB ID: {mongo_id}")
            
            logger.info(f"Search completed - found {len(valid_results)} valid results")
            return valid_distances, valid_results
            
        except Exception as e:
            logger.exception(f"FAISS search error: {e}")
            return [], []

    def save_index(self, filepath: str):
        """Save FAISS index to disk"""
        try:
            logger.debug(f"Saving index - total vectors: {self.index.ntotal}")
            faiss.write_index(self.index, filepath)
            logger.info(f"Saved FAISS index to {filepath}")
            
            # Save ID mapping
            logger.debug(f"ID mapping size: {len(self.id_mapping)}")
            
        except Exception as e:
            logger.exception(f"Error saving index: {e}")
            raise

    def load_index(self, filepath: str):
        """Load FAISS index from disk"""
        try:
            logger.debug(f"Loading index from {filepath}")
            self.index = faiss.read_index(str(filepath))
            logger.info(f"Loaded FAISS index - total vectors: {self.index.ntotal}")
            
        except Exception as e:
            logger.exception(f"Error loading index: {e}")
            raise

    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "mapped_ids": len(self.id_mapping),
            "next_id": self.next_id
        }

