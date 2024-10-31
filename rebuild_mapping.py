#!/usr/bin/env python

import os
import pickle
import faiss
from faiss_store import FAISSVectorStore  # Corrected import
from data_ingestion import DataIngestion  # Assuming this handles data loading and preprocessing

# Path to save the FAISS index mapping file
MAPPING_PATH = '/home/work/rag-faiss-embedding/data/faiss_index_mapping.pkl'
INDEX_PATH = '/home/work/rag-faiss-embedding/data/faiss_index.bin'

def rebuild_faiss_index():
    """
    Rebuilds the FAISS index from scratch and saves the index and mapping to disk.
    """
    # Step 1: Load data for indexing
    print("Loading data for indexing...")
    data_loader = DataIngestion()  # Adjust if DataIngestion requires parameters
    documents = data_loader.load_data()

    # Step 2: Initialize FAISS index
    print("Initializing FAISS index...")
    store = FAISSVectorStore()
    index = store.create_index()  # Make sure create_index initializes the correct FAISS index

    # Step 3: Add data to FAISS index
    print("Adding data to FAISS index...")
    id_to_doc_map = {}  # Dictionary to map FAISS IDs to document data
    for doc_id, doc in enumerate(documents):
        embedding = store.get_embedding(doc)  # Ensure this method retrieves the vector for indexing
        index.add(embedding)
        id_to_doc_map[doc_id] = doc

    # Step 4: Save the FAISS index and mapping
    print("Saving FAISS index and mapping...")
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, 'wb') as f:
        pickle.dump(id_to_doc_map, f)

    print("Rebuild complete. Index and mapping saved to disk.")

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(MAPPING_PATH), exist_ok=True)
    rebuild_faiss_index()

