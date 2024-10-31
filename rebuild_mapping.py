#!/usr/bin/env python3
# rebuild_mapping.py

import os
from database import Database
from vectorization import VectorizationPipeline
from loguru import logger
import sys

# Configure detailed logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add("logs/rebuild_mapping.log", level="DEBUG")

def rebuild_and_test():
    try:
        # Initialize components
        logger.info("Initializing components...")
        db = Database()
        
        # Print initial state
        doc_count = db.collection.count_documents({})
        vector_count = db.vector_store.index.ntotal
        logger.info(f"Initial state:")
        logger.info(f"- MongoDB documents: {doc_count}")
        logger.info(f"- FAISS vectors: {vector_count}")
        
        # Rebuild mapping
        logger.info("Rebuilding ID mapping...")
        success = db.rebuild_id_mapping()
        
        if not success:
            logger.error("Failed to rebuild mapping")
            return
        
        # Print final state
        vector_count = db.vector_store.index.ntotal
        mapping_count = len(db.vector_store.id_mapping)
        logger.info(f"\nFinal state:")
        logger.info(f"- MongoDB documents: {doc_count}")
        logger.info(f"- FAISS vectors: {vector_count}")
        logger.info(f"- ID mappings: {mapping_count}")
        
        # Test a search
        vectorizer = VectorizationPipeline()
        test_query = "python"
        logger.info(f"\nTesting search with query: '{test_query}'")
        
        # Generate embedding
        embedding = vectorizer.generate_embeddings([test_query])[0]
        
        # Get documents
        docs = db.get_similar_documents(embedding, top_k=3)
        
        if docs:
            logger.info(f"Found {len(docs)} documents:")
            for i, doc in enumerate(docs, 1):
                logger.info(f"Document {i}:")
                logger.info(f"- Title: {doc.get('title')}")
                logger.info(f"- Score: {doc.get('score', 0):.3f}")
                print(f"\n{i}. {doc.get('title')}")
                print(f"Score: {doc.get('score', 0):.3f}")
        else:
            logger.warning("No documents found")
            print("\nNo results found")

    except Exception as e:
        logger.exception("Error during rebuild and test")
        raise

if __name__ == "__main__":
    rebuild_and_test()

