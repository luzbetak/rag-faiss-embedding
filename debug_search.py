#!/usr/bin/env python3
# debug_search.py

import os
from database import Database
from vectorization import VectorizationPipeline
from loguru import logger
import sys

# Configure detailed logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add("logs/debug_search.log", level="DEBUG")

def debug_search():
    try:
        # Initialize components
        logger.info("Initializing components...")
        db = Database()
        vectorizer = VectorizationPipeline()
        
        # Print database stats
        doc_count = db.collection.count_documents({})
        vector_count = db.vector_store.index.ntotal
        logger.info(f"Database stats:")
        logger.info(f"- MongoDB documents: {doc_count}")
        logger.info(f"- FAISS vectors: {vector_count}")
        
        # Print sample document
        sample_doc = db.collection.find_one({})
        if sample_doc:
            logger.info("Sample document:")
            logger.info(f"- Title: {sample_doc.get('title')}")
            logger.info(f"- URL: {sample_doc.get('url')}")
        
        # Test searches
        test_queries = ["python", "algorithm", "installation", "linux"]
        
        for query in test_queries:
            logger.info(f"\nTesting search: '{query}'")
            
            # Generate embedding
            embedding = vectorizer.generate_embeddings([query])[0]
            logger.debug(f"Generated embedding - shape: {len(embedding)}")
            logger.debug(f"Embedding preview: {embedding[:5]}...")
            
            # Get FAISS search results
            distances, ids = db.vector_store.search(embedding, k=3)
            logger.debug(f"FAISS results - distances: {distances}")
            logger.debug(f"FAISS results - IDs: {ids}")
            
            # Get documents
            docs = db.get_similar_documents(embedding, top_k=3)
            
            if docs:
                logger.info(f"Found {len(docs)} documents:")
                for i, doc in enumerate(docs, 1):
                    logger.info(f"Document {i}:")
                    logger.info(f"- Title: {doc.get('title')}")
                    logger.info(f"- Score: {doc.get('score', 0):.3f}")
            else:
                logger.warning("No documents found")
            
            print(f"\nResults for '{query}':")
            if docs:
                for i, doc in enumerate(docs, 1):
                    print(f"\n{i}. {doc.get('title')}")
                    print(f"Score: {doc.get('score', 0):.3f}")
            else:
                print("No results found")
            
            input("\nPress Enter to continue...")

    except Exception as e:
        logger.exception("Debug search error")
        raise

if __name__ == "__main__":
    debug_search()

