# import chromadb
# from chromadb.config import Settings
# from typing import Dict, List, Optional
# from pathlib import Path

# def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
#     """Discover available ChromaDB backends in the project directory"""
#     backends = {}
#     current_dir = Path(".")
    
#     # Look for ChromaDB directories
#     # TODO: Create list of directories that match specific criteria (directory type and name pattern)

#     # TODO: Loop through each discovered directory
#         # TODO: Wrap connection attempt in try-except block for error handling
        
#             # TODO: Initialize database client with directory path and configuration settings
            
#             # TODO: Retrieve list of available collections from the database
            
#             # TODO: Loop through each collection found
#                 # TODO: Create unique identifier key combining directory and collection names
#                 # TODO: Build information dictionary containing:
#                     # TODO: Store directory path as string
#                     # TODO: Store collection name
#                     # TODO: Create user-friendly display name
#                     # TODO: Get document count with fallback for unsupported operations
#                 # TODO: Add collection information to backends dictionary
        
#         # TODO: Handle connection or access errors gracefully
#             # TODO: Create fallback entry for inaccessible directories
#             # TODO: Include error information in display name with truncation
#             # TODO: Set appropriate fallback values for missing information

#     # TODO: Return complete backends dictionary with all discovered collections

# def initialize_rag_system(chroma_dir: str, collection_name: str):
#     """Initialize the RAG system with specified backend (cached for performance)"""

#     # TODO: Create a chomadb persistentclient
#     # TODO: Return the collection with the collection_name

# def retrieve_documents(collection, query: str, n_results: int = 3, 
#                       mission_filter: Optional[str] = None) -> Optional[Dict]:
#     """Retrieve relevant documents from ChromaDB with optional filtering"""

#     # TODO: Initialize filter variable to None (represents no filtering)

#     # TODO: Check if filter parameter exists and is not set to "all" or equivalent
#     # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs

#     # TODO: Execute database query with the following parameters:
#         # TODO: Pass search query in the required format
#         # TODO: Set maximum number of results to return
#         # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)

#     # TODO: Return query results to caller

# def format_context(documents: List[str], metadatas: List[Dict]) -> str:
#     """Format retrieved documents into context"""
#     if not documents:
#         return ""
    
#     # TODO: Initialize list with header text for context section

#     # TODO: Loop through paired documents and their metadata using enumeration
#         # TODO: Extract mission information from metadata with fallback value
#         # TODO: Clean up mission name formatting (replace underscores, capitalize)
#         # TODO: Extract source information from metadata with fallback value  
#         # TODO: Extract category information from metadata with fallback value
#         # TODO: Clean up category name formatting (replace underscores, capitalize)
        
#         # TODO: Create formatted source header with index number and extracted information
#         # TODO: Add source header to context parts list
        
#         # TODO: Check document length and truncate if necessary
#         # TODO: Add truncated or full document content to context parts list

#     # TODO: Join all context parts with newlines and return formatted string




"""
RAG Client — Searches ChromaDB and formats context for Claude.

Flow:
  User question
       ↓
  Convert to vector (same model used in pipeline)
       ↓
  Find similar vectors in ChromaDB
       ↓
  Return matching NASA document chunks
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# MUST match the model used in embedding_pipeline.py!
# If they don't match → wrong vectors → bad search results
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """
    Scan current directory for ChromaDB folders.
    Returns dict so Streamlit UI can show a dropdown.
    """
    backends = {}
    current_dir = Path(".")

    # Find all folders with 'chroma' in the name
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and 'chroma' in d.name.lower()
    ]

    for chroma_dir in chroma_dirs:
        try:
            # Try connecting to this folder as ChromaDB
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # Get all collections inside this folder
            collections = client.list_collections()

            for collection in collections:
                # Unique key = folder::collection
                key = f"{chroma_dir.name}::{collection.name}"

                # Get document count safely
                try:
                    coll_obj = client.get_collection(collection.name)
                    doc_count = coll_obj.count()
                except Exception:
                    doc_count = 0

                # Store all info UI needs
                backends[key] = {
                    "directory":      str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name":   f"{collection.name} ({chroma_dir.name}) — {doc_count} docs",
                    "doc_count":      str(doc_count),
                }

        except Exception as e:
            # Can't connect — still show it with error label
            backends[chroma_dir.name] = {
                "directory":       str(chroma_dir),
                "collection_name": "",
                "display_name":    f"{chroma_dir.name} [Error: {str(e)[:40]}]",
                "doc_count":       "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, 
                           collection_name: str) -> Tuple:
    """
    Connect to ChromaDB collection and return it ready to search.
    
    IMPORTANT: We attach the SAME embedding function used in pipeline.
    Why? So when user types a question, ChromaDB automatically 
    converts it to a vector using the same model.
    
    Returns: (collection, True, "") on success
             (None, False, error_message) on failure
    """
    try:
        # Same embedding model as pipeline — MUST match!
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=DEFAULT_EMBEDDING_MODEL
        )

        # Connect to ChromaDB folder
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get the collection with embedding function attached
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )

        return collection, True, ""

    except Exception as e:
        return None, False, str(e)
    
def retrieve_documents(collection,
                       query: str,
                       n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """
    Search ChromaDB for chunks most relevant to the query.
    
    Args:
        collection    : connected ChromaDB collection
        query         : user's question in plain English
        n_results     : how many chunks to return (configurable)
        mission_filter: "apollo_11", "apollo_13", "challenger", or None
    
    Returns:
        ChromaDB result dict containing:
        - documents : the actual text chunks
        - metadatas : source, mission, category info
        - distances : how similar each chunk is (lower = better)
    """

    # No filter by default — search ALL missions
    where_filter = None

    # If user selected a specific mission in UI → filter by it
    # ChromaDB filter syntax: {"field": {"$eq": "value"}}
    if mission_filter and mission_filter.lower() not in ("all", ""):
        where_filter = {"mission": {"$eq": mission_filter}}

    # Execute semantic search
    # query_texts → ChromaDB auto-embeds using attached model
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    return results


def format_context(documents: List[str],
                   metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """
    Format retrieved chunks into clean readable context for Claude.
    
    WHY FORMATTING MATTERS:
    Claude needs clear source labels to cite correctly.
    Numbered sections make it easy to reference.
    Deduplication avoids sending same text twice.
    """
    if not documents:
        return ""

    context_parts = ["=== RELEVANT NASA DOCUMENTS ===\n"]
    seen_texts = []  # for deduplication

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):

        # Check for near-duplicate chunks — skip if >80% similar
        is_duplicate = False
        for seen in seen_texts:
            doc_words  = set(doc.lower().split())
            seen_words = set(seen.lower().split())
            if len(doc_words) > 0:
                overlap = len(doc_words & seen_words) / len(doc_words)
                if overlap > 0.8:
                    is_duplicate = True
                    break

        if is_duplicate:
            continue

        seen_texts.append(doc)

        # Clean up display names
        mission  = meta.get("mission", "Unknown").replace("_", " ").title()
        source   = meta.get("source", "Unknown")
        category = meta.get("document_category", "document")
        category = category.replace("_", " ").title()

        # Add relevance score if available
        score_str = ""
        if distances is not None and i < len(distances):
            score_str = f" | Score: {distances[i]:.3f}"

        # Build numbered header
        header = (f"[{i+1}] Mission: {mission} | "
                  f"Source: {source} | "
                  f"Type: {category}{score_str}")

        context_parts.append(header)
        context_parts.append("-" * 50)

        # Truncate very long chunks
        MAX_LEN = 800
        snippet = doc[:MAX_LEN] + "...[truncated]" if len(doc) > MAX_LEN else doc
        context_parts.append(snippet)
        context_parts.append("")  # blank line between chunks

    return "\n".join(context_parts)