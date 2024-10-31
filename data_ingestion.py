# data_ingestion.py
from typing import List, Dict, Any
import json
import pandas as pd
from loguru import logger
from database import Database

class DataIngestionPipeline:
    def __init__(self):  # Remove db_url parameter
        self.db = Database()  # Use MongoDB Database class
        logger.add("logs/pipeline.log")
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON or CSV files"""
        logger.info(f"Loading data from {file_path}")
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path).to_dict('records')
        else:
            raise ValueError("Unsupported file format")
        return data
    
    def preprocess_data(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and preprocess the data"""
        logger.info("Preprocessing data")
        processed_docs = []
        for doc in documents:
            # Basic text cleaning
            if 'content' in doc:
                doc['content'] = doc['content'].strip().lower()
            # Handle missing values
            doc = {k: v if v is not None else '' for k, v in doc.items()}
            processed_docs.append(doc)
        return processed_docs
    
    def store_documents(self, documents: List[Dict[str, Any]]):
        """Store documents in database"""
        logger.info("Storing documents in database")
        self.db.batch_store_documents(documents)
