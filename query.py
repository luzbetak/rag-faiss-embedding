#!/usr/bin/env python3

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from typing import List, Optional, Dict
from database import Database
from vectorization import VectorizationPipeline
from transformers import pipeline
from config import Config
import numpy as np

# Set OpenBLAS environment variables
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OPENBLAS_MAIN_FREE': '1',
    'OMP_NUM_THREADS': '1'
})

# Global QueryEngine instance
query_engine = None

class SearchRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3

class SearchResponse(BaseModel):
    similar_documents: List[Dict]
    generated_response: str

class QueryEngine:
    def __init__(self):
        """Initialize the query engine with necessary components"""
        try:
            # Initialize components
            self.db = Database()
            self.vectorization = VectorizationPipeline()
            
            # Load FAISS index if exists
            faiss_path = Config.FAISS_INDEX_PATH
            if faiss_path.exists():
                logger.info(f"Loading FAISS index from {faiss_path}")
                self.db.load_vector_store(str(faiss_path))
                vector_count = self.db.vector_store.index.ntotal
                logger.info(f"FAISS index loaded with {vector_count} vectors")
            else:
                logger.error(f"FAISS index not found at {faiss_path}")
                raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

            # Initialize text generation
            logger.info("Initializing text generation model...")
            self.generator = pipeline('text2text-generation', 
                                   model='google/flan-t5-base', 
                                   max_length=200)
            
            # Verify initialization
            doc_count = self.db.collection.count_documents({})
            vector_count = self.db.vector_store.index.ntotal
            
            logger.info("Query engine initialization complete:")
            logger.info(f"- MongoDB documents: {doc_count}")
            logger.info(f"- FAISS vectors: {vector_count}")
            logger.info(f"- Vector dimension: {Config.VECTOR_DIMENSION}")
            
            if doc_count == 0 or vector_count == 0:
                logger.warning("Database or FAISS index is empty!")
                
        except Exception as e:
            logger.exception("Error initializing query engine")
            raise

    async def verify_system_state(self) -> bool:
        """Verify system state before search"""
        try:
            doc_count = self.db.collection.count_documents({})
            vector_count = self.db.vector_store.index.ntotal
            
            if doc_count == 0:
                logger.error("MongoDB collection is empty")
                return False
                
            if vector_count == 0:
                logger.error("FAISS index is empty")
                return False
                
            if doc_count != vector_count:
                logger.warning(f"Document count mismatch: MongoDB={doc_count}, FAISS={vector_count}")
                
            return True
            
        except Exception as e:
            logger.exception("Error verifying system state")
            return False

    async def search(self, query: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """Perform vector similarity search"""
        try:
            # Verify system state
            if not await self.verify_system_state():
                logger.error("System state verification failed")
                return []
            
            # Log search request
            logger.info(f"Processing search query: '{query}' (top_k={top_k})")
            
            # Generate embedding
            query_embedding = self.vectorization.generate_embeddings([query])[0]
            logger.info(f"Generated embedding vector (dim={len(query_embedding)})")
            
            # Perform search
            similar_docs = self.db.get_similar_documents(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Log results
            if similar_docs:
                logger.info(f"Found {len(similar_docs)} similar documents:")
                for i, doc in enumerate(similar_docs, 1):
                    logger.info(f"- Doc {i}: {doc.get('title', 'N/A')} "
                              f"(score: {doc.get('score', 0):.3f})")
            else:
                logger.warning("No similar documents found")
            
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
            logger.info("Response generation successful")
            
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for the FastAPI application"""
    global query_engine
    
    try:
        # Startup
        logger.info("Initializing RAG Search API...")
        query_engine = QueryEngine()
        logger.info("RAG Search API initialization complete")
        
        yield
        
    except Exception as e:
        logger.exception("Error during API initialization")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down RAG Search API...")
        if query_engine:
            query_engine.close()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Search API",
    description="API for vector similarity search and response generation",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        doc_count = query_engine.db.collection.count_documents({})
        vector_count = query_engine.db.vector_store.index.ntotal
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "documents_count": doc_count,
            "vectors_count": vector_count,
            "vector_dimension": Config.VECTOR_DIMENSION
        }
    except Exception as e:
        logger.exception("Health check error")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search endpoint that performs vector similarity search and generates a response"""
    try:
        # Log the incoming request
        logger.info(f"Search request received: {request.text}")
        
        # Perform vector similarity search
        similar_docs = await query_engine.search(request.text, top_k=request.top_k)
        
        # Generate a response based on similar documents
        generated_response = await query_engine.generate_response(
            query=request.text,
            documents=similar_docs
        )
        
        # Log response stats
        logger.info(f"Returning response with {len(similar_docs)} documents")
        
        return SearchResponse(
            similar_documents=similar_docs,
            generated_response=generated_response
        )
        
    except Exception as e:
        logger.exception(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

