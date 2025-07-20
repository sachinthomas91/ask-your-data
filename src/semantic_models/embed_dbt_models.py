# src/semantic_models/embed_dbt_models.py

import os
import sys
import yaml
import hashlib
import requests
import chromadb
from chromadb.config import Settings
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = str(PROJECT_ROOT / ".chroma_store")
COLLECTION_NAME = "dbt_schema_models"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

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
    print(f"[ERROR] ChromaDB initialization failed: {e}")
    raise

def ollama_embed(text: str) -> list:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def process_schema_yml(filepath: str):
    with open(filepath, "r") as f:
        content = yaml.safe_load(f)

    model_entries = content.get("models", [])
    entries = []

    for model in model_entries:
        name = model.get("name", "unknown_model")
        description = model.get("description", "")
        columns = model.get("columns", [])

        column_text = "\n".join([
            f"- {col['name']}: {col.get('description', '')}" for col in columns
        ])

        combined_text = f"""
Model: {name}
Description: {description}

Columns:
{column_text}
        """.strip()

        entries.append({
            "text": combined_text,
            "model_name": name,
            "file_path": filepath
        })

    return entries

def embed_models_from_folder(folder_path: str):
    count = 0
    found_files = 0
    errors = []

    # Verify ChromaDB is ready
    try:
        collection.count()
    except Exception as e:
        print(f"[ERROR] ChromaDB collection is not accessible: {e}")
        return

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
                        except Exception as e:
                            error_msg = f"Error adding document {doc_id}: {e}"
                            print(f"[ERROR] {error_msg}")
                            errors.append(error_msg)
                            
                except Exception as e:
                    error_msg = f"Error processing {full_path}: {e}"
                    print(f"[ERROR] {error_msg}")
                    errors.append(error_msg)

    if found_files == 0:
        print(f"[WARN] No .yml or .yaml files found in {folder_path}")
    
    print(f"✅ Embedded and stored {count} dbt models.")
    
    # Report final status
    if errors:
        print("\n[WARN] The following errors occurred during processing:")
        for error in errors:
            print(f"  - {error}")
            
    # Verify final state
    if not os.path.isdir(DB_PATH):
        print(f"[ERROR] ChromaDB store directory was NOT created at: {DB_PATH}")
    elif not os.listdir(DB_PATH):
        print("[WARN] ChromaDB store directory is empty")

# Optional CLI entry point
if __name__ == "__main__":
    target_folder = os.path.join("dbt_ask_data", "models", "marts")
    embed_models_from_folder(target_folder)
