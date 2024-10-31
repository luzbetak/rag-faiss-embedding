#!/usr/bin/env python3
# query.py

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

# Set OpenBLAS environment variables
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OPENBLAS_MAIN_FREE': '1',
    'OMP_NUM_THREADS': '1'
})

# Global QueryEngine instance
query_engine = None

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

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Search API",
    description="API for vector similarity search and response generation",
    version="1.0.0",
    lifespan=lifespan
)

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
            # Generate query embedding
            query_embedding = self.vectorization.generate_embeddings([query])[0]
            
            # Log the search request
            logger.info(f"Searching for: {query}")
            
            # Get similar documents using FAISS/MongoDB hybrid search
            similar_docs = self.db.get_similar_documents(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            logger.info(f"Found {len(similar_docs)} similar documents")
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    async def generate_response(self, query: str, documents: List[Dict]) -> str:
        """Generate a response based on the query and retrieved documents"""
        try:
            if not documents:
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
            
            # Generate response
            response = self.generator(prompt)[0]['generated_text']
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return "I apologize, but I encountered an error generating a response."

    def close(self):
        """Cleanup resources"""
        self.db.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
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
        
        # Log the response size
        logger.info(f"Returning {len(similar_docs)} documents with response")
        
        return SearchResponse(
            similar_documents=similar_docs,
            generated_response=generated_response
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

