import streamlit as st
import chromadb
from chromadb.config import Settings
import requests
import os
import json
import pathlib
import pandas as pd
import psycopg2
import re
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- Load Environment Variables ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    print(f"[INFO] Loading environment variables from: {ENV_PATH}")
    load_dotenv(ENV_PATH)
else:
    print("[WARN] No .env file found in project root")

# --- Constants ---
DB_PATH = str(PROJECT_ROOT / ".chroma_store")
COLLECTION_NAME = "dbt_schema_models"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "qwen2.5-coder:7b" # "qwen2:7b" # "mistral"
OLLAMA_BASE_URL = "http://localhost:11434/api"

# --- Verify ChromaDB store ---
if not os.path.isdir(DB_PATH):
    st.error(f"ChromaDB store not found at {DB_PATH}. Please run the embedding script first.")
    st.stop()

# --- Initialize ChromaDB ---
try:
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"[INFO] Found collection '{COLLECTION_NAME}' with {collection.count()} documents")
except Exception as e:
    st.error(f"Error connecting to ChromaDB: {e}")
    st.stop()

# --- Ollama Utilities ---
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

# --- Query & Context Utilities ---
def get_relevant_context(query: str, collection, n_results: int = 3) -> tuple[str, List[Dict[str, Any]]]:
    query_embedding = ollama_embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    if not documents:
        return "", []
    
    context_parts = [f"Model: {meta['model_name']}\n{doc}" for doc, meta in zip(documents, metadatas)]
    return "\n\n".join(context_parts), metadatas

def is_sql_query_request(query: str) -> bool:
    sql_keywords = ["sql", "query", "select", "from", "write a query", "show me the query"]
    return any(keyword.lower() in query.lower() for keyword in sql_keywords)

def generate_sql_query(query: str, context: str) -> str:
    system_prompt = """
        - Write clean, well-structured, and readable SQL.
        - Reference dbt models using the dwh.analytics_dev. prefix.
        - Use CTEs to organize logic; avoid subqueries unless absolutely necessary
        - Avoid JOINs unless combining data from multiple tables is required.
        - Prefer single-table queries when all necessary data exists in one source.
        - Return only the SQL query. Any explanations or context should be in inline comments.
        - Ensure correctness, simplicity, and performance optimization. Keep queries as short and minimal as possible.
        - Never use ";" at the end of the query.
    """
    return ollama_chat_completion(query, system=system_prompt, context=context)

def answer_question(query: str, context: str) -> str:
    system_prompt = """You are a helpful data warehouse expert. Using the provided dbt model documentation,
    answer the user's question accurately and concisely. If you're not sure about something, say so.
    Focus on explaining the data models and their relationships."""
    
    return ollama_chat_completion(query, system=system_prompt, context=context)

# --- Database Utilities ---
def get_postgres_conn():
    required_vars = ["POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = (
            "Missing required environment variables:\n"
            f"{', '.join(missing_vars)}\n"
            "Please check your .env file in the project root."
        )
        st.error(error_msg)
        return None
        
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to PostgreSQL: {e}")
        return None

def execute_query(query: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame"""
    try:
        conn = get_postgres_conn()
        if not conn:
            return pd.DataFrame()
        
        # Remove semicolon if it exists at the end of the query
        query = query.strip()
        if query.endswith(';'):
            query = query[:-1].strip()
            
        # Add LIMIT if not present
        if "LIMIT" not in query.upper():
            query = f"{query} LIMIT 20"
            
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            results = cur.fetchall()
            return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Query execution failed: {e}")
        return pd.DataFrame()

def extract_sql_from_markdown(text: str) -> str:
    """Extract SQL query from text that might contain markdown or other content."""
    sql_match = re.search(r"```sql\n(.*?)```", text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    return text.strip()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Ask Your dbt Models", layout="wide")
st.title("Ask Your dbt Models (Natural Language Query)")

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
    st.session_state.current_query = None
    st.session_state.current_sql = None
    st.session_state.current_results = None
    st.session_state.metadatas = None
    st.session_state.context = None
    st.session_state.answer = None

# Query Input
query = st.text_input("Ask a question about your dbt models or request a SQL query:", 
                     "Which models have a foreign key to customers?")
ask_clicked = st.button("Ask")

# Process Query
if ask_clicked:
    with st.spinner("Thinking..."):
        try:
            context, metadatas = get_relevant_context(query, collection)
            st.session_state.current_query = query
            st.session_state.metadatas = metadatas
            st.session_state.context = context

            if is_sql_query_request(query):
                sql_query = generate_sql_query(query, context)
                st.session_state.current_sql = sql_query
                st.session_state.current_results = None
                st.session_state.answer = None
            else:
                answer = answer_question(query, context)
                st.session_state.answer = answer
                st.session_state.current_sql = None
                st.session_state.current_results = None

        except Exception as e:
            st.error(f"Error: {e}")
            print(f"[ERROR] Query failed: {str(e)}")

# Display Results in Chat-like Interface
if st.session_state.current_query:
    st.markdown("---")
    st.markdown(f"### 🤔 Question")
    st.markdown(f"_{st.session_state.current_query}_")
    
    # Display Answer or SQL based on query type
    if st.session_state.answer:
        st.markdown("### 🤖 Answer")
        st.write(st.session_state.answer)
    
    if st.session_state.current_sql:
        st.markdown("### 📝 Generated SQL")
        st.code(st.session_state.current_sql, language="sql")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            execute_clicked = st.button("▶️ Execute Query", key=f"execute_sql_{hash(st.session_state.current_query)}")

        if execute_clicked:
            with st.spinner("Executing query..."):
                clean_sql = extract_sql_from_markdown(st.session_state.current_sql)
                results_df = execute_query(clean_sql)
                st.session_state.current_results = results_df

        # Display Results
        if st.session_state.current_results is not None:
            df = st.session_state.current_results
            if not df.empty:
                st.markdown("### 📊 Query Results")
                st.dataframe(df, use_container_width=True, height=400)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "query_results.csv",
                    "text/csv",
                    key=f"download_csv_{hash(st.session_state.current_query)}"
                )
            else:
                st.info("No results returned from the query.")
    
    # Display Referenced Models
    if st.session_state.metadatas:
        st.markdown("### 📚 Referenced Models")
        for meta in st.session_state.metadatas:
            with st.expander(f"📊 {meta['model_name']}"):
                st.markdown(f"**Model:** `{meta['model_name']}`")
                st.markdown(f"**File:** `{meta['file_path']}`")
    
    # Optional: Display Raw Context in an expander
    if st.session_state.context:
        with st.expander("🔍 Show Raw Context"):
            st.code(st.session_state.context)
    
    st.markdown("---")
