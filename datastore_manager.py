#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import json
from loguru import logger
from database import Database
from data_ingestion import DataIngestionPipeline
from vectorization import VectorizationPipeline
from config import Config
import numpy as np

# Control threading and suppress warnings
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OPENBLAS_MAIN_FREE': '1',
    'OMP_NUM_THREADS': '1'
})

# Configure logging using settings from Config
logger.remove()
logger.add(sys.stderr, format=Config.LOG_FORMAT)
logger.add(Config.LOG_FILE, rotation=Config.LOG_ROTATION)

class RAGDatabaseInitializer:
    def __init__(self):
        """Initialize RAG database manager"""
        try:
            self.db = Database()
            self.data_pipeline = DataIngestionPipeline()
            self.vectorization_pipeline = VectorizationPipeline()
            logger.info("RAG Database Manager initialized successfully")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def verify_search_index(self) -> bool:
        """Verify search index file exists and contains valid data"""
        if not Config.SEARCH_INDEX_PATH.exists():
            logger.error(f"Search index not found: {Config.SEARCH_INDEX_PATH}")
            return False
            
        try:
            with open(Config.SEARCH_INDEX_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    logger.error("Empty search index")
                    return False
                logger.info(f"Found {len(data)} documents in search index")
                return True
        except Exception as e:
            logger.error(f"Error reading search index: {e}")
            return False

    def init_database(self) -> bool:
        """Initialize database with required indices"""
        logger.info("Initializing database...")
        try:
            # Drop existing collections
            self.db.collection.drop()
            
            # Create MongoDB indices
            self.db.collection.create_index([("url", 1)], unique=True)
            self.db.collection.create_index([("title", 1)])
            
            # Initialize new FAISS index
            vector_dim = self.vectorization_pipeline.model.get_sentence_embedding_dimension()
            self.db.init_vector_store(dimension=vector_dim)
            
            # Verify initialization
            count = self.db.collection.count_documents({})
            logger.info(f"Database initialized with {count} documents")
            logger.info("Database initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Database initialization error: {e}", exc_info=True)
            return False

    def load_documents(self) -> bool:
        """Load and process documents from search index"""
        if not self.verify_search_index():
            return False

        try:
            # Load and preprocess documents
            documents = self.data_pipeline.load_data(str(Config.SEARCH_INDEX_PATH))
            processed_docs = self.data_pipeline.preprocess_data(documents)
            
            if not processed_docs:
                logger.error("No documents to process")
                return False

            # Generate embeddings
            embeddings = self.vectorization_pipeline.generate_embeddings(
                [doc["content"] for doc in processed_docs]
            )
            
            # Store documents and embeddings
            self.db.batch_store_documents(processed_docs, embeddings)
            
            # Verify storage
            count = self.db.collection.count_documents({})
            logger.info(f"Loaded {count} documents successfully!")
            return True

        except Exception as e:
            logger.error(f"Document loading error: {e}", exc_info=True)
            return False

    def save_indices(self) -> bool:
        """Save FAISS index to disk"""
        try:
            self.db.save_vector_store(str(Config.FAISS_INDEX_PATH))
            logger.info(f"Indices saved to {Config.FAISS_INDEX_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error saving indices: {e}", exc_info=True)
            return False

    def load_indices(self) -> bool:
        """Load FAISS index from disk"""
        try:
            if not Config.FAISS_INDEX_PATH.exists():
                logger.error(f"FAISS index file not found: {Config.FAISS_INDEX_PATH}")
                return False
                
            self.db.load_vector_store(str(Config.FAISS_INDEX_PATH))
            logger.info(f"Indices loaded from {Config.FAISS_INDEX_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error loading indices: {e}", exc_info=True)
            return False

    def verify_system(self) -> bool:
        """Verify system integrity by performing a test search"""
        try:
            # Get document count
            doc_count = self.db.collection.count_documents({})
            logger.info(f"Documents in MongoDB: {doc_count}")
            
            if doc_count == 0:
                logger.warning("No documents found in database")
                return False
            
            # Get a sample document
            sample_doc = self.db.collection.find_one({})
            if not sample_doc:
                logger.error("Could not retrieve sample document")
                return False
            
            logger.info(f"Testing with document: {sample_doc.get('title', 'Untitled')}")
            
            # Generate test embedding
            sample_content = sample_doc.get('content', '')[:1000]  # Use first 1000 chars
            test_embedding = self.vectorization_pipeline.generate_embeddings([sample_content])[0]
            
            # Perform similarity search
            results = self.db.get_similar_documents(test_embedding, top_k=3)
            
            if not results:
                logger.error("Similarity search returned no results")
                return False
            
            # Log search results
            logger.info(f"Found {len(results)} similar documents:")
            for i, doc in enumerate(results, 1):
                logger.info(f"{i}. {doc.get('title', 'Untitled')} (score: {doc.get('score', 0):.3f})")
            
            logger.info("System verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System verification error: {e}", exc_info=True)
            return False

    def test_similarity_search(self) -> bool:
        """Test similarity search with a query"""
        try:
            # Get user input
            query = input("\nEnter search query: ")
            top_k = int(input("Number of results (default 3): ") or "3")
            
            # Generate embedding for query
            query_embedding = self.vectorization_pipeline.generate_embeddings([query])[0]
            
            # Search for similar documents
            results = self.db.get_similar_documents(query_embedding, top_k=top_k)
            
            if not results:
                print("\nNo similar documents found")
                return False
                
            # Display results
            print(f"\nFound {len(results)} similar documents:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc.get('title', 'Untitled')}")
                print(f"Score: {doc.get('score', 0):.3f}")
                print(f"URL: {doc.get('url', 'N/A')}")
                print(f"Content preview: {doc.get('content', 'N/A')[:200]}...")
                
            return True
            
        except Exception as e:
            logger.error(f"Search test error: {e}")
            return False

def print_document_count(db: Database):
    """Print current document count and sample document"""
    try:
        count = db.collection.count_documents({})
        print(f"\nTotal documents in database: {count}")
        if count > 0:
            sample = db.collection.find_one({}, {'_id': 0})
            print("\nSample document:")
            print(json.dumps(sample, indent=2))
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        print("\nError retrieving document count")

def main():
    try:
        # Create instance of RAG database initializer
        rag_init = RAGDatabaseInitializer()
        
        # Define menu options
        menu_options = {
            "1": ("Initialize database (will delete existing data)", rag_init.init_database),
            "2": ("Load documents from search index", rag_init.load_documents),
            "3": ("Save indices to disk", rag_init.save_indices),
            "4": ("Load indices from disk", rag_init.load_indices),
            "5": ("Verify system integrity", rag_init.verify_system),
            "6": ("Show document count", lambda: print_document_count(rag_init.db)),
            "7": ("Test similarity search", rag_init.test_similarity_search),
            "8": ("Exit", sys.exit)
        }

        print("\nRAG System Database Initialization")
        print("=================================")

        while True:
            try:
                # Display menu
                print("\nOptions:")
                for key, (description, _) in menu_options.items():
                    print(f"{key}. {description}")

                # Get user choice
                choice = input("\nEnter your choice (1-8): ")
                
                if choice in menu_options:
                    # Execute chosen operation
                    result = menu_options[choice][1]()
                    if result is not None:
                        print("Operation " + ("succeeded" if result else "failed"))
                    if choice != "8":
                        input("\nPress Enter to continue...")
                else:
                    print("\nInvalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in menu operation: {e}", exc_info=True)
                print(f"\nError: {e}")
                input("\nPress Enter to continue...")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        try:
            rag_init.db.close()
        except:
            pass

if __name__ == "__main__":
    main()

