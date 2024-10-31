# vectorization.py
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from config import Config
from database import Database
from loguru import logger

class VectorizationPipeline:
    def __init__(self):
        self.model = SentenceTransformer(Config.MODEL_NAME)
        self.db = Database()
        logger.info("Initialized vectorization pipeline")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=True
        )
        return embeddings.tolist()
    
    def process_documents(self, documents: List[Dict[str, Any]]):
        """Process documents and store with embeddings"""
        texts = [doc.get('content', '') for doc in documents]
        embeddings = self.generate_embeddings(texts)
        self.db.batch_store_documents(documents, embeddings)

