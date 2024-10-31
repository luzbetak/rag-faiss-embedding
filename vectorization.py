# vectorization.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from loguru import logger
from typing import List
from tqdm import tqdm

class VectorizationPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("Initialized vectorization pipeline")

    @torch.no_grad()
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            outputs = self.model(**encoded)
            
            # Use CLS token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

