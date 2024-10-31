# query.py

from loguru import logger
from typing import List, Dict
from database import Database
from vectorization import VectorizationPipeline
from transformers import pipeline
import numpy as np

class QueryEngine:
    def __init__(self):
        """Initialize the query engine with necessary components"""
        self.db = Database()
        self.vectorization = VectorizationPipeline()
        self.generator = pipeline('text2text-generation', 
                                model='google/flan-t5-base', 
                                max_length=200)
        logger.info("Query engine initialized")

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform vector similarity search"""
        try:
            # Log the search request
            logger.info(f"Processing search query: {query}")
            
            # Verify database connection
            doc_count = self.db.get_document_count()
            logger.info(f"Total documents in database: {doc_count}")
            
            # Generate query embedding
            logger.info("Generating query embedding...")
            query_embedding = self.vectorization.generate_embeddings([query])[0]
            logger.info(f"Generated embedding of length: {len(query_embedding)}")
            
            # Convert embedding to numpy array
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Get similar documents using FAISS
            logger.info(f"Searching for similar documents with top_k={top_k}")
            distances, indices = self.db.vector_store.search(query_embedding, top_k)
            
            # Fetch documents from SQLite
            similar_docs = []
            for idx, distance in zip(indices, distances):
                # Add 1 to idx since SQLite IDs start at 1
                doc = self.db.get_document_by_id(int(idx) + 1)
                if doc:
                    # Convert distance to similarity score (1 / (1 + distance))
                    doc["score"] = float(1.0 / (1.0 + distance))
                    similar_docs.append(doc)
                    logger.info(f"Found document {doc['id']} with score {doc['score']:.3f}")
            
            if not similar_docs:
                logger.warning("No similar documents found")
            else:
                logger.info(f"Found {len(similar_docs)} similar documents")
            
            return similar_docs
            
        except Exception as e:
            logger.exception(f"Search error: {str(e)}")
            return []

    async def generate_response(self, query: str, documents: List[Dict]) -> str:
        """Generate a response based on the query and retrieved documents"""
        try:
            if not documents:
                logger.info("No documents available for response generation")
                return "No relevant documents found to answer your query."
                
            # Format context from documents
            context_parts = []
            for i, doc in enumerate(documents, 1):
                score = doc.get('score', 0.0)
                title = doc.get('title', 'Unknown')
                content = doc.get('content', '')
                context_parts.append(
                    f"Document {i} (Score: {score:.3f}, Title: {title}):\n{content}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Create prompt
            prompt = (
                f"Based on the following documents, answer this question: {query}\n\n"
                f"Context:\n{context}\n\n"
                f"Answer:"
            )
            
            logger.info("Generating response...")
            response = self.generator(prompt)[0]['generated_text']
            logger.info("Response generated successfully")
            
            return response.strip()
            
        except Exception as e:
            logger.exception(f"Response generation error: {str(e)}")
            return "I apologize, but I encountered an error generating a response."

    def close(self):
        """Cleanup resources"""
        try:
            self.db.close()
            logger.info("Query engine resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
