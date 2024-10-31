# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Base Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME = "rag_database"
    COLLECTION_NAME = "documents"
    
    # Model Configuration
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 32
    VECTOR_DIMENSION = 384
    
    # FAISS Configuration
    FAISS_INDEX_TYPE = "L2"  # Options: "L2" or "IP" (Inner Product)
    FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
    
    # Search Configuration
    TOP_K = 5
    
    # Data Files
    SEARCH_INDEX_PATH = DATA_DIR / "search-index.json"
    
    # Logging Configuration
    LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    LOG_FILE = LOGS_DIR / "rag_system.log"
    LOG_ROTATION = "500 MB"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        dirs = [cls.DATA_DIR, cls.LOGS_DIR]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def get_mongodb_uri(cls):
        """Get MongoDB URI with database name"""
        return f"{cls.MONGODB_URI}/{cls.DATABASE_NAME}"
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        # Check if model name is valid
        if not cls.MODEL_NAME:
            raise ValueError("MODEL_NAME must be specified")
            
        # Check vector dimension
        if cls.VECTOR_DIMENSION <= 0:
            raise ValueError("VECTOR_DIMENSION must be positive")
            
        # Check FAISS index type
        if cls.FAISS_INDEX_TYPE not in ["L2", "IP"]:
            raise ValueError("FAISS_INDEX_TYPE must be either 'L2' or 'IP'")
            
        # Check batch size
        if cls.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be positive")
            
        # Check TOP_K
        if cls.TOP_K <= 0:
            raise ValueError("TOP_K must be positive")
            
        return True

# Setup directories on import
Config.setup_directories()

# Validate configuration on import
try:
    Config.validate_config()
except Exception as e:
    raise ValueError(f"Configuration validation failed: {str(e)}")    
