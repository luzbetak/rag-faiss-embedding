# vectorization.py
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from config import Config
from loguru import logger

class VectorizationPipeline:
    def __init__(self):
        self.model = SentenceTransformer(Config.MODEL_NAME)
        logger.info("Initialized vectorization pipeline")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=Config.BATCH_SIZE, show_progress_bar=True)
        return embeddings.tolist()

    def process_and_store_documents(self, documents: List[Dict[str, Any]], db_instance):
        texts = [doc.get('content', '') for doc in documents]
        embeddings = self.generate_embeddings(texts)
        db_instance.batch_store_documents(documents, embeddings)

