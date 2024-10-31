#!/usr/bin/env python3

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import argparse
from pymongo import MongoClient
from datetime import datetime
from json import JSONEncoder

# Configure OpenBLAS to avoid warnings
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from bs4 import BeautifulSoup, Tag
import spacy
from spacy.language import Language

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 512  # Maximum content length in characters
MAX_SENTENCES = 2        # Maximum number of sentences for summary

class DateTimeEncoder(JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class IndexEntry:
    """Data class for storing content data"""
    _id_counter = 1  # Class variable to keep track of incremental IDs

    def __init__(self, url: str, title: str, content: str):
        self.id = IndexEntry._id_counter  # Assign the current counter value as the ID
        IndexEntry._id_counter += 1       # Increment the counter for the next entry
        self.url = url
        self.title = title
        self.content = content[:MAX_CONTENT_LENGTH] if content else ""  # Limit content length
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        """Convert entry to dictionary with ISO format dates"""
        return {
            'id': self.id,
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class TextSummarizer:
    def __init__(self, output_dir: str = "data", mongodb_uri: str = "mongodb://localhost:27017/"):
        """Initialize the text summarizer with spaCy NLP and MongoDB connection"""
        self.output_dir = Path(output_dir).resolve()
        self.nlp = self._initialize_spacy()
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client.rag_database
        self.collection = self.db.documents
        logger.info(f"Initialized TextSummarizer with output directory: {self.output_dir}")
        logger.info(f"Connected to MongoDB database: {self.db.name}")

    def _initialize_spacy(self) -> Language:
            """Initialize spaCy with error handling"""
            try:
                # Try to load the medium model first (has word vectors)
                try:
                    nlp = spacy.load("en_core_web_md")
                    if nlp.has_vectors:
                        logger.info("Loaded spaCy medium model with word vectors")
                    else:
                        logger.warning("Loaded model does not have word vectors")
                except OSError:
                    logger.warning("SpaCy model 'en_core_web_md' not found. Installing...")
                    os.system("python -m spacy download en_core_web_md")
                    nlp = spacy.load("en_core_web_md")
                    logger.info("Installed and loaded spaCy medium model")
    
                # Add sentencizer if not already present
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                    logger.info("Added sentencizer to pipeline")
    
                return nlp
    
            except Exception as e:
                logger.error(f"Error loading spaCy model: {e}")
                logger.warning("Falling back to small model without word vectors")
                
                try:
                    nlp = spacy.load("en_core_web_sm")
                    if "sentencizer" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")
                    return nlp
                except Exception as e:
                    logger.error(f"Failed to load fallback model: {e}")
                    raise

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove common HTML-related words
        text = re.sub(r'\b(menu|html|title|include|nav|header|footer)\b', '', text, flags=re.IGNORECASE)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?-]', ' ', text)
        # Replace multiple dashes with space
        text = re.sub(r'-+', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple periods with single period
        text = re.sub(r'\.+', '.', text)
        return text.strip()

    def extract_key_sentences(self, doc) -> List[str]:
            """Extract key sentences based on position and length"""
            sentences = list(doc.sents)
            if not sentences:
                return []
    
            # Always include the first sentence if it's long enough
            key_sentences = []
            if sentences and len(sentences[0].text.split()) >= 3:
                key_sentences.append(sentences[0])
    
            # Add additional sentences based on length and content
            for sent in sentences[1:]:
                # Skip very short sentences
                if len(sent.text.split()) < 3:
                    continue
                    
                # Check similarity only if model has vectors
                if hasattr(self.nlp, 'has_vectors') and self.nlp.has_vectors:
                    try:
                        # Skip sentences that are too similar to what we already have
                        if any(sent.similarity(existing) > 0.7 for existing in key_sentences):
                            continue
                    except Exception as e:
                        logger.debug(f"Similarity check failed: {e}")
                        # If similarity check fails, fall back to length-based selection
                        pass
                
                key_sentences.append(sent)
                if len(key_sentences) >= MAX_SENTENCES:
                    break
    
            return [sent.text.strip() for sent in key_sentences]

    def extract_text_from_html(self, soup: BeautifulSoup) -> str:
        """Extract text from HTML while preserving important content"""
        # First, store all <pre> tags content
        pre_tags = soup.find_all('pre')
        pre_contents = [tag.extract() for tag in pre_tags]  # Remove and save pre tags
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Get main content
        content_areas = soup.find_all(['main', 'article', 'section'])
        if content_areas:
            text = ' '.join(area.get_text(separator=' ', strip=True) 
                          for area in content_areas)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Skip summarization for pre-formatted content
        pre_texts = '\n'.join(pre.get_text() for pre in pre_contents)
        
        return f"{text}\n{pre_texts}" if pre_texts else text

    def summarize_text(self, text: str) -> str:
        """Summarize the text using spaCy"""
        try:
            if not text.strip():
                return ""
                
            doc = self.nlp(text)
            key_sentences = self.extract_key_sentences(doc)
            summary = ' '.join(key_sentences)
            
            # Ensure summary is within length limit
            if len(summary) > MAX_CONTENT_LENGTH:
                # Try to truncate at a sentence boundary
                summary = summary[:MAX_CONTENT_LENGTH]
                last_period = summary.rfind('.')
                if last_period > 0:
                    summary = summary[:last_period + 1]
            
            return summary.strip()
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:MAX_CONTENT_LENGTH]

    def process_html_file(self, file_path: Path) -> Optional[IndexEntry]:
        """Process a single HTML file and return a content entry"""
        try:
            logger.debug(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
                # Extract text while preserving pre tags
                extracted_text = self.extract_text_from_html(soup)
                clean_text = self.clean_text(extracted_text)

                if not clean_text:
                    logger.warning(f"Skipping {file_path}: No meaningful content")
                    return None

                content = self.summarize_text(clean_text)
                relative_path = str(file_path.relative_to(Path.cwd()))
                url_path = f"https://kevinluzbetak.com/{relative_path}"

                return IndexEntry(
                    url=url_path.strip(),
                    title=file_path.name,
                    content=content
                )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def _write_to_mongodb(self, entries: List[IndexEntry]) -> None:
        """Write entries to MongoDB"""
        try:
            documents = [entry.to_dict() for entry in entries if entry and entry.url and entry.title and entry.content]
            
            if not documents:
                logger.error("No valid entries to write to MongoDB!")
                return

            self.collection.delete_many({})
            result = self.collection.insert_many(documents)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} documents into MongoDB")
            
            self.collection.create_index([("content", "text")])
            logger.info("Created text index on content field")

        except Exception as e:
            logger.error(f"Error writing to MongoDB: {e}", exc_info=True)
            raise

    def _write_index_file(self, entries: List[IndexEntry]) -> None:
        """Write the summaries to a JSON file"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.output_dir / "documents.json"
            
            valid_entries = [
                entry.to_dict() for entry in entries
                if entry and entry.url and entry.title and entry.content
            ]

            if not valid_entries:
                logger.error("No valid entries to write!")
                return

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(valid_entries, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
            logger.info(f"Successfully created {output_file} with {len(valid_entries)} entries")

        except Exception as e:
            logger.error(f"Error writing index file: {e}", exc_info=True)
            raise

    def generate_index(self) -> None:
        """Generate the content index from HTML files"""
        logger.info("Starting content index generation...")

        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")

        html_files = [
            p for p in current_dir.rglob("*.html")
            if p.name != "index.html" and not str(p).startswith(str(self.output_dir))
        ]

        if not html_files:
            logger.warning("No HTML files found to process")
            return

        logger.info(f"Found {len(html_files)} HTML files to process")
        logger.debug(f"Files to process: {[str(f) for f in html_files]}")

        with ThreadPoolExecutor() as executor:
            entries = list(filter(None, executor.map(
                self.process_html_file,
                html_files
            )))

        if not entries:
            logger.error("No valid entries generated from HTML files")
            return

        logger.info(f"Generated {len(entries)} valid entries")

        self._write_to_mongodb(entries)
        self._write_index_file(entries)

        logger.info("Index generation completed")

    def __del__(self):
        """Cleanup MongoDB connection"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate content index from HTML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for the content index"
    )
    parser.add_argument(
        "--mongodb-uri",
        default="mongodb://localhost:27017/",
        help="MongoDB connection URI"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--max-content-length",
        type=int,
        default=512,
        help="Maximum length of content in characters"
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=2,
        help="Maximum number of sentences in summary"
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    global MAX_CONTENT_LENGTH, MAX_SENTENCES
    MAX_CONTENT_LENGTH = args.max_content_length
    MAX_SENTENCES = args.max_sentences

    try:
        summarizer = TextSummarizer(args.output_dir, args.mongodb_uri)
        summarizer.generate_index()
    except Exception as e:
        logger.error(f"Failed to generate content index: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
