#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data
Uses FREE local sentence-transformers — no OpenAI key needed.
"""

import os
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Configure logging — writes to BOTH terminal AND a log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    """
    Reads NASA .txt files → chunks them → embeds with local model → stores in ChromaDB
    """

    def __init__(self,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 #embedding_model: str = "all-MiniLM-L6-v2", #getting OutofMemeory Exception every time so changing to another model
                 embedding_model: str = "paraphrase-MiniLM-L3-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):

        # 1. Store all config as instance variables so other methods can use them
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 2. Create the FREE local embedding function
        #    This downloads all-MiniLM-L6-v2 on first run (~80MB), then caches it
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # 3. Connect to ChromaDB — creates the folder if it doesn't exist
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 4. Get or create a collection inside ChromaDB
        #    Think of a collection like a "table" in a database
        #    colllection ===table 
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Collection '{collection_name}' ready. "
                    f"Documents in it: {self.collection.count()}")
        

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
            """
            Split a large text into overlapping chunks.
            Each chunk gets its own copy of metadata + its position info.
            """

            # CASE 1: Short text — no chunking needed, return as-is
            if len(text) <= self.chunk_size:
                chunk_meta = {**metadata, "chunk_index": 0, "total_chunks": 1}
                return [(text, chunk_meta)]

            chunks = []
            start = 0
            chunk_index = 0

            while start < len(text):

                # Calculate where this chunk ends
                end = start + self.chunk_size

                # CASE 2: We're not at the end of the text yet
                # Try to end at a sentence boundary (". ") for cleaner chunks
                if end < len(text):
                    sentence_end = text.rfind('. ', start, end)

                    if sentence_end > start + (self.chunk_size // 2):
                        # Found a sentence boundary in the second half — use it
                        end = sentence_end + 1
                    else:
                        # No good sentence boundary — fall back to word boundary
                        word_end = text.rfind(' ', start, end)
                        if word_end > start:
                            end = word_end

                # Extract the chunk and strip extra whitespace
                chunk = text[start:end].strip()

                if chunk:  # Only add non-empty chunks
                    chunk_meta = {
                        **metadata,           # copy all original metadata
                        "chunk_index": chunk_index,
                        "chunk_start": start,
                        "chunk_end": end,
                    }
                    chunks.append((chunk, chunk_meta))
                    chunk_index += 1

                # Move start forward — but go BACK by chunk_overlap for the next chunk
                # This is what creates the overlapping effect
                start = end - self.chunk_overlap

                # Safety: stop if we've gone past the end
                if start >= len(text):
                    break

            # Now we know total_chunks — add it to every chunk's metadata
            total = len(chunks)
            chunks = [(c, {**m, "total_chunks": total}) for c, m in chunks]

            return chunks

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Create a stable unique ID for each chunk.
        Format: apollo_13__AS13_TEC__chunk_0001
        
        WHY stable? If you run the pipeline twice, same chunk = same ID.
        This lets us SKIP already-processed chunks (no duplicates).
        """
        mission = metadata.get("mission", "unknown")
        source = metadata.get("source", file_path.stem)
        chunk_index = metadata.get("chunk_index", 0)

        # Remove special characters from source name that would break IDs
        safe_source = "".join(
            c if c.isalnum() or c == "_" else "_"
            for c in source
        )[:50]  # max 50 chars

        return f"{mission}__{safe_source}__chunk_{chunk_index:04d}"
        #                                               ↑↑↑↑
        #                              :04d means pad with zeros → 0001, 0002, 0099
        
    def check_document_exists(self, doc_id: str) -> bool:
        """
        Ask ChromaDB: does this ID already exist?
        Used to skip re-processing during update_mode='skip'
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result["ids"]) > 0
        except Exception:
            return False
        
        
    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get ALL chunk IDs that came from one specific file.
        Used in update_mode='replace' to delete old chunks before re-adding.
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            all_docs = self.collection.get()

            return [
                all_docs['ids'][i]
                for i, meta in enumerate(all_docs['metadatas'])
                if meta.get('source') == source and meta.get('mission') == mission
            ]
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []
    
    
    def extract_mission_from_path(self, file_path: Path) -> str:
        """
        Look at the folder name to determine which mission this file belongs to.
        
        data_text/apollo11/something.txt  → 'apollo_11'
        data_text/apollo13/something.txt  → 'apollo_13'
        data_text/challenger/something.txt → 'challenger'
        """
        path_str = str(file_path).lower()

        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """
        Look at the file path to determine what KIND of data this is.
        Transcript? Flight plan? Audio recording?
        """
        path_str = str(file_path).lower()

        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'

    def extract_document_category_from_filename(self, filename: str) -> str:
        """
        Look at the FILENAME itself to categorize the document.
        
        a11transcript_PAO → public_affairs_officer (press/public communications)
        AS13_CM           → command_module (cockpit communications)
        AS13_TEC          → technical (engineering data)
        """
        f = filename.lower()

        if 'pao' in f:         return 'public_affairs_officer'
        elif 'cm' in f:        return 'command_module'
        elif 'tec' in f:       return 'technical'
        elif 'flight_plan' in f: return 'flight_plan'
        elif 'mission_audio' in f: return 'mission_audio'
        elif 'ntrs' in f:      return 'nasa_archive'
        elif '19900066485' in f: return 'technical_report'
        elif '19710015566' in f: return 'mission_report'
        elif 'full_text' in f: return 'complete_document'
        else:                  return 'general_document'
        
    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read one .txt file → attach metadata → chunk it.
        Returns list of (chunk_text, metadata) tuples.
        """
        try:
            # Read the raw file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            # Skip empty files
            if not content.strip():
                logger.warning(f"Empty file skipped: {file_path.name}")
                return []

            # Build metadata dict for this file
            # This same metadata gets copied into EVERY chunk from this file
            metadata = {
                'source':    file_path.stem,        # filename without extension
                'file_path': str(file_path),
                'file_type': 'text',
                'mission':   self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }

            logger.info(f"  mission={metadata['mission']} | "
                       f"category={metadata['document_category']} | "
                       f"size={metadata['file_size']} chars")

            # Hand off to chunk_text — returns list of (text, meta) tuples
            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []




    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Walk through data_text/ and collect all .txt files.
        Skips hidden files and summary files.
        """
        base_path = Path(base_path)
        files_to_process = []

        # These are the three mission folders we expect
        data_dirs = ['apollo11', 'apollo13', 'challenger']

        for data_dir in data_dirs:
            dir_path = base_path / data_dir

            if dir_path.exists():
                txt_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(txt_files)
                logger.info(f"Found {len(txt_files)} files in {data_dir}")
            else:
                logger.warning(f"Directory not found: {dir_path}")

        # Filter out unwanted files
        # filtered = [
        #     f for f in files_to_process
        #     if not f.name.startswith('.')
        #     and 'summary' not in f.name.lower()
        #     and f.suffix.lower() == '.txt'
        # ]
        
        # Files that are too large for your RAM - skip these by name
        SKIP_FILES = [
            '19900066485_textract_full_text.txt',
            'Apollo_11_Flight_Plan_HSK_textract_full_text.txt',
            'NASA_NTRS_Archive_19710015566_textract_full_text.txt',
            'AS13_TEC_textract_full_text.txt',
            'a11transcript_tec_textract_full_text.txt',
        ]

        filtered = [
            f for f in files_to_process
            if not f.name.startswith('.')
            and 'summary' not in f.name.lower()
            and f.suffix.lower() == '.txt'
            and f.name not in SKIP_FILES      # ← skip problem files by name
        ]

        logger.info(f"Total files to process: {len(filtered)}")
        return filtered
    
    
    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]],
                                    file_path: Path, batch_size: int = 50,
                                    update_mode: str = 'skip') -> Dict[str, int]:
        """
        Save chunks into ChromaDB with 3 modes:
        
        skip    → if chunk ID already exists, ignore it (DEFAULT)
        update  → if chunk ID already exists, overwrite it
        replace → delete ALL chunks from this file first, then re-add
        """
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}

        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        # REPLACE MODE: wipe all existing chunks from this file first
        if update_mode == 'replace':
            existing_ids = self.get_file_documents(file_path)
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(f"Deleted {len(existing_ids)} existing chunks")

        # Prepare batch containers
        batch_ids, batch_docs, batch_metas = [], [], []

        for text, metadata in documents:

            # Generate the stable ID for this chunk
            doc_id = self.generate_document_id(file_path, metadata)

            # SKIP MODE: already exists? move on
            if update_mode == 'skip' and self.check_document_exists(doc_id):
                stats['skipped'] += 1
                continue

            # UPDATE MODE: already exists? overwrite it
            if update_mode == 'update' and self.check_document_exists(doc_id):
                self.collection.update(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata]
                )
                stats['updated'] += 1
                continue

            # NEW chunk — stage it for batch add
            batch_ids.append(doc_id)
            batch_docs.append(text)
            batch_metas.append(metadata)

            # When batch is full — flush it to ChromaDB
            if len(batch_ids) >= batch_size:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                )
                stats['added'] += len(batch_ids)
                logger.info(f"    Saved batch of {len(batch_ids)} chunks")
                # Reset batch containers
                batch_ids, batch_docs, batch_metas = [], [], []

        # Flush any remaining chunks that didn't fill a full batch
        if batch_ids:
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
            stats['added'] += len(batch_ids)

        return stats
    def process_all_text_data(self, base_path: str, 
                               update_mode: str = 'skip') -> Dict[str, Any]:
        """
        Master method — processes ALL NASA files end to end.
        Calls every method we built in the right order.
        """
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}          # breakdown per mission
        }

        # Step 1: Find all files
        files = self.scan_text_files_only(base_path)

        # Step 2: Process each file one by one
        for file_path in files:
            logger.info(f"\nProcessing: {file_path.name}")
            mission = self.extract_mission_from_path(file_path)

            try:
                # Step 3: Read + chunk the file
                documents = self.process_text_file(file_path)

                if not documents:
                    logger.warning(f"No content from {file_path.name}")
                    continue

                # Step 4: Save chunks to ChromaDB
                file_stats = self.add_documents_to_collection(
                    documents, file_path, update_mode=update_mode
                )

                # Step 5: Update overall stats
                stats['files_processed'] += 1
                stats['documents_added'] += file_stats['added']
                stats['documents_updated'] += file_stats['updated']
                stats['documents_skipped'] += file_stats['skipped']
                stats['total_chunks'] += len(documents)

                # Step 6: Update per-mission breakdown
                if mission not in stats['missions']:
                    stats['missions'][mission] = {
                        'files': 0, 'chunks': 0,
                        'added': 0, 'updated': 0, 'skipped': 0
                    }
                stats['missions'][mission]['files'] += 1
                stats['missions'][mission]['chunks'] += len(documents)
                stats['missions'][mission]['added'] += file_stats['added']
                stats['missions'][mission]['skipped'] += file_stats['skipped']

                logger.info(f"chunks={len(documents)} | "
                           f"added={file_stats['added']} | "
                           f"skipped={file_stats['skipped']}")

            except Exception as e:
                logger.error(f"Error on {file_path}: {e}")
                stats['errors'] += 1

        return stats
    
    def get_collection_info(self) -> Dict[str, Any]:
        """How many documents are in our collection right now?"""
        return {
            'collection_name': self.collection_name,
            'document_count':  self.collection.count(),
            'persist_directory': self.chroma_persist_directory,
            'embedding_model': self.embedding_model,
        }
        
        

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Test search — type a question, get back matching chunks.
        ChromaDB automatically embeds query_text using our local model.
        """
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {}
def main():
    parser = argparse.ArgumentParser(
        description='NASA Embedding Pipeline - FREE local embeddings'
    )
    parser.add_argument('--data-path',       default='.',
                        help='Path to data_text folder')
    parser.add_argument('--chroma-dir',      default='./chroma_db',
                        help='Where to save ChromaDB')
    parser.add_argument('--collection-name', default='nasa_space_missions_text')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--chunk-size',      type=int, default=500)
    parser.add_argument('--chunk-overlap',   type=int, default=100)
    parser.add_argument('--batch-size',      type=int, default=50)
    parser.add_argument('--update-mode',
                        choices=['skip', 'update', 'replace'],
                        default='skip')
    parser.add_argument('--test-query',      help='Run a test search after processing')
    parser.add_argument('--stats-only',      action='store_true',
                        help='Just show collection stats, no processing')
    args = parser.parse_args()

    # Create pipeline
    pipeline = ChromaEmbeddingPipelineTextOnly(
        chroma_persist_directory = args.chroma_dir,
        collection_name          = args.collection_name,
        embedding_model          = args.embedding_model,
        chunk_size               = args.chunk_size,
        chunk_overlap            = args.chunk_overlap
    )

    # Stats only mode
    if args.stats_only:
        info = pipeline.get_collection_info()
        logger.info(f"Collection : {info['collection_name']}")
        logger.info(f"Documents  : {info['document_count']}")
        logger.info(f"Model      : {info['embedding_model']}")
        return

    # Run full pipeline
    start_time = time.time()
    stats = pipeline.process_all_text_data(
        args.data_path, update_mode=args.update_mode
    )
    elapsed = time.time() - start_time

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed : {stats['files_processed']}")
    logger.info(f"Total chunks    : {stats['total_chunks']}")
    logger.info(f"Added           : {stats['documents_added']}")
    logger.info(f"Skipped         : {stats['documents_skipped']}")
    logger.info(f"Errors          : {stats['errors']}")
    logger.info(f"Time taken      : {elapsed:.1f} seconds")
    logger.info("\nPer mission:")
    for mission, m in stats['missions'].items():
        logger.info(f"  {mission}: {m['files']} files, "
                   f"{m['chunks']} chunks, {m['added']} added")

    # Optional test query
    if args.test_query:
        logger.info(f"\nTest query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            for i, doc in enumerate(results['documents'][0][:3]):
                logger.info(f"Result {i+1}: {doc[:200]}...")


if __name__ == "__main__":
    main()