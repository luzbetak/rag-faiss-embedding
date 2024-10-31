#!/usr/bin/env python3

import os
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
        self.db = Database()
        self.vectorization = VectorizationPipeline()
        self.generator = pipeline('text2text-generation', 
                                model='google/flan-t5-base', 
                                max_length=200)
        logger.info("Query engine initialized")

    async def search(self, query: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """Perform vector similarity search"""
        try:
            # Log the search request
            logger.info(f"Processing search query: {query}")
            
            # Verify database connection
            doc_count = self.db.collection.count_documents({})
            logger.info(f"Total documents in database: {doc_count}")
            
            # Generate query embedding
            logger.info("Generating query embedding...")
            query_embedding = self.vectorization.generate_embeddings([query])[0]
            logger.info(f"Generated embedding of length: {len(query_embedding)}")
            
            # Get similar documents
            logger.info(f"Searching for similar documents with top_k={top_k}")
            similar_docs = self.db.get_similar_documents(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            if similar_docs:
                logger.info(f"Found {len(similar_docs)} similar documents")
                for i, doc in enumerate(similar_docs, 1):
                    logger.info(f"Document {i} - Score: {doc.get('score', 0):.3f}, "
                              f"Title: {doc.get('title', 'N/A')}")
            else:
                logger.warning("No similar documents found")
            
            return similar_docs
            
        except Exception as e:
            logger.exception(f"Search error: {str(e)}")
            raise

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

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for the FastAPI application"""
    global query_engine
    
    # Startup
    logger.info("Initializing RAG Search API...")
    query_engine = QueryEngine()
    
    yield
    
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
    doc_count = query_engine.db.collection.count_documents({})
    return {
        "status": "healthy",
        "version": "1.0.0",
        "documents_count": doc_count
    }

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

