#!/usr/bin/env python3
# initialize_rag.py

import os
import sqlite3
from pathlib import Path
from loguru import logger
from typing import List, Dict
import numpy as np
from database import Database
from vectorization import VectorizationPipeline
import glob

def process_python_files(directory: str = ".") -> List[Dict]:
    """Process Python files in the directory and return as documents"""
    documents = []
    for python_file in glob.glob(f"{directory}/**/*.py", recursive=True):
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
                rel_path = os.path.relpath(python_file, directory)
                documents.append({
                    "url": rel_path,
                    "title": os.path.basename(python_file),
                    "content": content
                })
                logger.info(f"Processed file: {rel_path}")
        except Exception as e:
            logger.error(f"Error processing {python_file}: {e}")
    return documents

def main():
    # Initialize components
    db = Database()
    vectorizer = VectorizationPipeline()
    
    # Process Python files
    documents = process_python_files()
    logger.info(f"Found {len(documents)} Python files")
    
    # Insert documents into SQLite
    db.insert_documents(documents)
    
    # Generate embeddings
    contents = [doc["content"] for doc in documents]
    embeddings = vectorizer.generate_embeddings(contents)
    
    # Get document IDs from SQLite
    doc_ids = []
    for doc in documents:
        db.cursor.execute('SELECT id FROM documents WHERE url = ?', (doc["url"],))
        result = db.cursor.fetchone()
        if result:
            doc_ids.append(result[0])
    
    # Add embeddings to FAISS
    db.vector_store.reset()  # Clear existing index
    db.vector_store.add_vectors(embeddings, doc_ids)
    
    # Save the FAISS index
    db.vector_store.save_index()
    
    logger.info(f"Initialized RAG system with {len(documents)} documents")
    db.close()

if __name__ == "__main__":
    main()
