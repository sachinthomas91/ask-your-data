import streamlit as st
import chromadb
from chromadb.config import Settings
import requests
import os
import json
from typing import List, Dict, Any
import re
import pathlib
# Constants (reuse from embedding script)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = str(PROJECT_ROOT / ".chroma_store")
COLLECTION_NAME = "dbt_schema_models"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "mistral" # "qwen2:7b"  # or any other model you prefer
OLLAMA_BASE_URL = "http://localhost:11434/api"

print(f"[INFO] Looking for ChromaDB store at: {DB_PATH}")

# Verify ChromaDB store exists
if not os.path.isdir(DB_PATH):
    st.error(f"ChromaDB store not found at {DB_PATH}. Please run the embedding script first.")
    st.stop()

# Initialize Chroma client with PersistentClient
try:
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    print(f"[INFO] Successfully connected to ChromaDB store")
    
    # Get collection and verify it exists
    collection = client.get_collection(name=COLLECTION_NAME)
    count = collection.count()
    print(f"[INFO] Found collection '{COLLECTION_NAME}' with {count} documents")
    
except Exception as e:
    st.error(f"Error connecting to ChromaDB: {e}")
    st.stop()

def ollama_embed(text: str) -> list:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def ollama_chat_completion(prompt: str, system: str | None = None, context: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if context:
        messages.append({"role": "user", "content": f"Here is some relevant context:\n{context}"})
    messages.append({"role": "user", "content": prompt})
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/chat",
        json={
            "model": OLLAMA_CHAT_MODEL,
            "messages": messages,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

def get_relevant_context(query: str, collection, n_results: int = 3) -> tuple[str, List[Dict[str, Any]]]:
    query_embedding = ollama_embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    if not documents:
        return "", []
    
    # Format context from retrieved documents
    context_parts = []
    for doc, meta in zip(documents, metadatas):
        context_parts.append(f"Model: {meta['model_name']}\n{doc}")
    
    return "\n\n".join(context_parts), metadatas

def is_sql_query_request(query: str) -> bool:
    sql_keywords = ["sql", "query", "select", "from", "write a query", "show me the query"]
    return any(keyword.lower() in query.lower() for keyword in sql_keywords)

def generate_sql_query(query: str, context: str) -> str:
    system_prompt = """
    - Your query must be well-structured, cleanly formatted, and easy to read.
    - Always use CTEs (Common Table Expressions) to organize complex logic. Avoid sub-queries unless absolutely unavoidable.
    - Do not include JOINs unless data genuinely comes from multiple tables. If all necessary columns are in a single table, avoid JOINs entirely.
    - Output only the SQL query, with inline comments explaining the logic. Do not include any natural language before or after the query, and do not use markdown code blocks (e.g., no triple backticks).
    - Ensure the query is syntactically correct and optimized for performance.
    """
    
    return ollama_chat_completion(query, system=system_prompt, context=context)

def answer_question(query: str, context: str) -> str:
    system_prompt = """You are a helpful data warehouse expert. Using the provided dbt model documentation,
    answer the user's question accurately and concisely. If you're not sure about something, say so.
    Focus on explaining the data models and their relationships."""
    
    return ollama_chat_completion(query, system=system_prompt, context=context)

st.set_page_config(page_title="Ask Your dbt Models", layout="wide")
st.title("Ask Your dbt Models (Natural Language Query)")

query = st.text_input("Ask a question about your dbt models or request a SQL query:", 
                     "Which models have a foreign key to customers?")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        try:
            # Get relevant context from ChromaDB
            context, metadatas = get_relevant_context(query, collection)
            
            if not context:
                st.warning("I couldn't find any relevant information in the available models.")
                st.stop()
            
            # Check if this is a SQL query request
            if is_sql_query_request(query):
                st.subheader("Generated SQL Query")
                sql_query = generate_sql_query(query, context)
                st.code(sql_query, language="sql")
                
                st.subheader("Referenced Models")
                for meta in metadatas:
                    with st.expander(f"📊 {meta['model_name']}"):
                        st.markdown(f"**Model:** `{meta['model_name']}`")
                        st.markdown(f"**File:** `{meta['file_path']}`")
            else:
                # Generate natural language answer
                answer = answer_question(query, context)
                st.markdown("### Answer")
                st.write(answer)
                
                # Show referenced models in expandable sections
                st.markdown("### Referenced Models")
                for meta in metadatas:
                    with st.expander(f"📊 {meta['model_name']}"):
                        st.markdown(f"**Model:** `{meta['model_name']}`")
                        st.markdown(f"**File:** `{meta['file_path']}`")
                        
        except Exception as e:
            st.error(f"Error: {e}")
            print(f"[ERROR] Query failed: {str(e)}")
