# src/semantic_models/embed_dbt_models.py

import os
import sys
import yaml
import hashlib
import requests
import chromadb
import logging
import time
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(PROJECT_ROOT / ".chroma_store")
COLLECTION_NAME = "dbt_schema_models"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT = 30
MAX_TEXT_LENGTH = 2000  # Maximum characters for embedding (conservative to avoid Ollama crashes)

# Ensure ChromaDB directory exists
os.makedirs(DB_PATH, exist_ok=True)

try:
    # Initialize Chroma client with explicit settings
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "DBT schema model embeddings"}
    )
except Exception as e:
    logger.error(f"ChromaDB initialization failed: {e}")
    raise

def _truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Truncate text to maximum length while preserving structure.

    Args:
        text: Text to truncate
        max_length: Maximum length in characters

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    # Truncate and add indicator
    truncated = text[:max_length - 10] + "... [truncated]"
    logger.debug(f"Text truncated from {len(text)} to {len(truncated)} characters")
    return truncated

def _check_ollama_availability() -> bool:
    """Check if Ollama service is available."""
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )
        response.raise_for_status()
        return True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False

def ollama_embed(text: str, max_retries: int = 2) -> List[float]:
    """
    Get embeddings from Ollama for a single text with retry logic.

    Args:
        text: Text to embed
        max_retries: Maximum number of retry attempts

    Returns:
        List of embedding values

    Raises:
        requests.exceptions.RequestException: If Ollama request fails after retries
    """
    # Truncate text to safe length
    text = _truncate_text(text)

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=OLLAMA_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Failed to connect to Ollama at {OLLAMA_BASE_URL}. "
                "Ensure Ollama is running with: ollama serve"
            )
            raise
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(
                    f"Ollama request timed out (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying in 2 seconds..."
                )
                time.sleep(2)
                continue
            else:
                logger.error(f"Ollama request timed out after {OLLAMA_REQUEST_TIMEOUT}s")
                raise
        except requests.exceptions.HTTPError as e:
            if response.status_code == 500:
                logger.warning(
                    f"Ollama 500 error (attempt {attempt + 1}/{max_retries + 1}). "
                    f"This may indicate the model crashed or ran out of memory. "
                    f"Text length: {len(text)} chars. Retrying..."
                )
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                else:
                    logger.error(
                        f"Ollama server error after {max_retries + 1} attempts. "
                        f"Try restarting Ollama: ollama serve"
                    )
                    raise
            else:
                logger.error(f"Ollama HTTP error {response.status_code}: {e}")
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise

def process_schema_yml(filepath: str) -> List[Dict[str, str]]:
    """
    Parse a dbt schema.yml file and extract model information.

    Args:
        filepath: Path to the schema.yml file

    Returns:
        List of dictionaries containing model text, name, and file path

    Raises:
        yaml.YAMLError: If YAML parsing fails
        ValueError: If file doesn't contain expected dbt schema structure
    """
    try:
        with open(filepath, "r") as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in {filepath}: {e}")
        return []

    if not content or not isinstance(content, dict):
        logger.warning(f"File {filepath} does not contain valid YAML structure")
        return []

    model_entries = content.get("models", [])
    if not model_entries:
        logger.debug(f"No models found in {filepath}")
        return []

    entries = []

    for model in model_entries:
        # Validate model has required fields
        if not isinstance(model, dict) or "name" not in model:
            logger.warning(f"Skipping invalid model entry in {filepath}")
            continue

        name = model.get("name", "unknown_model")
        description = model.get("description", "")
        columns = model.get("columns", []) or []

        # Build column descriptions safely
        column_text = ""
        if columns:
            col_lines = []
            for col in columns:
                if isinstance(col, dict) and "name" in col:
                    col_name = col["name"]
                    col_desc = col.get("description", "")
                    col_lines.append(f"- {col_name}: {col_desc}")
            column_text = "\n".join(col_lines)

        combined_text = f"""Model: {name}
Description: {description}

Columns:
{column_text}""".strip()

        entries.append({
            "text": combined_text,
            "model_name": name,
            "file_path": filepath
        })

    return entries

def embed_models_from_folder(folder_path: str) -> Dict[str, int]:
    """
    Process all dbt schema files in a folder and embed them using Ollama.

    Args:
        folder_path: Path to the folder containing dbt schema files

    Returns:
        Dictionary with statistics: {"embedded": count, "errors": error_count, "files": found_files}
    """
    count = 0
    found_files = 0
    error_count = 0
    errors = []

    # Verify folder exists
    if not os.path.isdir(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return {"embedded": 0, "errors": 1, "files": 0}

    # Verify ChromaDB is ready
    try:
        collection.count()
    except Exception as e:
        logger.error(f"ChromaDB collection is not accessible: {e}")
        return {"embedded": 0, "errors": 1, "files": 0}

    # Check Ollama availability before processing
    if not _check_ollama_availability():
        logger.error(
            f"Ollama is not available at {OLLAMA_BASE_URL}. "
            "Start Ollama with: ollama serve"
        )
        return {"embedded": 0, "errors": 1, "files": 0}

    logger.info(f"Starting to process dbt models from {folder_path}")

    # Collect all existing document IDs for duplicate detection
    existing_docs = set()
    try:
        existing = collection.get()
        existing_docs = set(existing.get("ids", []))
    except Exception as e:
        logger.warning(f"Could not retrieve existing documents: {e}")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".yml", ".yaml")):
                found_files += 1
                full_path = os.path.join(root, file)
                try:
                    model_chunks = process_schema_yml(full_path)
                    for chunk in model_chunks:
                        uid = f"{chunk['file_path']}::{chunk['model_name']}"
                        doc_id = hashlib.md5(uid.encode()).hexdigest()

                        # Skip if already embedded
                        if doc_id in existing_docs:
                            logger.debug(f"Skipping already embedded model: {chunk['model_name']}")
                            continue

                        try:
                            embedding = ollama_embed(chunk["text"])
                            collection.add(
                                documents=[chunk["text"]],
                                metadatas=[{
                                    "file_path": chunk["file_path"],
                                    "model_name": chunk["model_name"]
                                }],
                                ids=[doc_id],
                                embeddings=[embedding]
                            )
                            count += 1
                            logger.debug(f"Embedded model: {chunk['model_name']}")
                        except Exception as e:
                            error_msg = f"Error adding document {doc_id}: {e}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                            error_count += 1

                except Exception as e:
                    error_msg = f"Error processing {full_path}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    error_count += 1

    # Report results
    logger.info(f"Processing complete: {found_files} files found, {count} models embedded, {error_count} errors")

    if found_files == 0:
        logger.warning(f"No .yml or .yaml files found in {folder_path}")

    if count > 0:
        logger.info(f"✅ Successfully embedded and stored {count} dbt models.")

    # Report errors
    if errors:
        logger.warning(f"Encountered {error_count} error(s) during processing:")
        for error in errors[:10]:  # Show first 10 errors
            logger.warning(f"  - {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")

    # Verify final state
    if not os.path.isdir(DB_PATH):
        logger.error(f"ChromaDB store directory was NOT created at: {DB_PATH}")
    elif not os.listdir(DB_PATH):
        logger.warning("ChromaDB store directory is empty")

    return {"embedded": count, "errors": error_count, "files": found_files}

# Optional CLI entry point
if __name__ == "__main__":
    # Use absolute path based on PROJECT_ROOT
    target_folder = str(PROJECT_ROOT / "dbt_ask_data" / "models" / "marts")

    if not os.path.isdir(target_folder):
        logger.error(f"Target folder does not exist: {target_folder}")
        sys.exit(1)

    results = embed_models_from_folder(target_folder)

    # Exit with error code if there were issues
    sys.exit(0 if results["errors"] == 0 else 1)
