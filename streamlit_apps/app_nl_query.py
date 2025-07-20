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
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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

# --- Visualization Utilities ---
def analyze_dataframe_for_viz(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze DataFrame to determine best visualization options"""
    if df.empty:
        return {"can_visualize": False, "reason": "Empty dataset"}
    
    analysis = {
        "can_visualize": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": {},
        "suggested_charts": []
    }
    
    # Analyze each column
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "unique_count": df[col].nunique(),
            "null_count": df[col].isnull().sum(),
            "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
            "is_datetime": pd.api.types.is_datetime64_any_dtype(df[col]),
            "is_categorical": False
        }
        
        # Improved categorical detection logic
        if not col_info["is_numeric"] and not col_info["is_datetime"]:
            # String/object columns are categorical
            col_info["is_categorical"] = True
        elif 'id' in col.lower() and col_info["unique_count"] < len(df):
            # ID columns (like seller_id, customer_id) should be categorical even if numeric
            col_info["is_categorical"] = True
        elif col_info["is_numeric"] and col_info["unique_count"] <= 10 and df[col].dtype in ['int64', 'int32'] and col_info["unique_count"] < len(df) * 0.3:
            # Only very small integer columns with few unique values (like status codes, categories)
            # But NOT floating point numbers which are likely continuous values
            col_info["is_categorical"] = True
        elif not col_info["is_numeric"] and col_info["unique_count"] < len(df) * 0.5:
            # Non-numeric columns with less than 50% unique values
            col_info["is_categorical"] = True
        
        analysis["columns"][col] = col_info
    
    # Determine suggested visualizations
    numeric_cols = [col for col, info in analysis["columns"].items() if info["is_numeric"]]
    categorical_cols = [col for col, info in analysis["columns"].items() if info["is_categorical"]]
    datetime_cols = [col for col, info in analysis["columns"].items() if info["is_datetime"]]
    
    # Always suggest at least one visualization if we have data
    if len(numeric_cols) >= 2:
        analysis["suggested_charts"].append("scatter")
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        analysis["suggested_charts"].append("bar")
        analysis["suggested_charts"].append("box")
    if len(categorical_cols) >= 1:
        analysis["suggested_charts"].append("pie")
    if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        analysis["suggested_charts"].append("line")
    if len(numeric_cols) >= 1:
        analysis["suggested_charts"].append("histogram")
    
    # Fallback: if no charts suggested but we have columns, suggest basic visualizations
    if not analysis["suggested_charts"]:
        if len(df.columns) >= 2:
            # Try to create a basic bar chart with first two columns
            analysis["suggested_charts"].append("bar")
        if len(numeric_cols) >= 1:
            analysis["suggested_charts"].append("histogram")
    
    return analysis

def create_visualization(df: pd.DataFrame, chart_type: str, analysis: Dict[str, Any]) -> go.Figure | None:
    """Create visualization based on data analysis"""
    try:
        numeric_cols = [col for col, info in analysis["columns"].items() if info["is_numeric"]]
        categorical_cols = [col for col, info in analysis["columns"].items() if info["is_categorical"]]
        datetime_cols = [col for col, info in analysis["columns"].items() if info["is_datetime"]]
        
        if chart_type == "bar" and categorical_cols and numeric_cols:
            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}")
            
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            color_col = categorical_cols[0] if categorical_cols else None
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=color_col,
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
            
        elif chart_type == "line" and datetime_cols and numeric_cols:
            fig = px.line(df, x=datetime_cols[0], y=numeric_cols[0],
                         title=f"{numeric_cols[0]} over time")
            
        elif chart_type == "pie" and categorical_cols:
            # Find numeric columns even more aggressively
            all_numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            # Try to convert object columns to numeric if they contain numbers
            potential_numeric_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        # Remove any currency symbols, commas, etc.
                        cleaned_series = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                        if not numeric_series.isna().all():  # If at least some values convert successfully
                            potential_numeric_cols.append(col)
                            # Update the dataframe with converted values
                            df[col] = numeric_series
                    except:
                        continue
            
            # Combine all numeric columns
            all_numeric_cols.extend(potential_numeric_cols)
            value_col = numeric_cols[0] if numeric_cols else (all_numeric_cols[0] if all_numeric_cols else None)
            
            # Store debug info for later display
            debug_info = {
                "original_numeric_cols": [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                "converted_cols": potential_numeric_cols,
                "all_numeric_cols": all_numeric_cols,
                "detected_numeric_cols": numeric_cols,
                "selected_value_col": value_col,
                "pie_chart_created": False,
                "using_counts": False
            }
            
            if value_col:
                # Show the actual data being used
                pie_data = df[[categorical_cols[0], value_col]].dropna()
                debug_info.update({
                    "names_col": categorical_cols[0],
                    "values_col": value_col,
                    "pie_data": pie_data,
                    "value_column_values": pie_data[value_col].tolist(),
                    "value_column_dtype": str(pie_data[value_col].dtype),
                    "name_column_values": pie_data[categorical_cols[0]].tolist()
                })
                
                if not pie_data.empty and pd.api.types.is_numeric_dtype(pie_data[value_col]):
                    # Try creating pie chart with explicit parameters
                    fig = px.pie(
                        pie_data, 
                        names=categorical_cols[0], 
                        values=value_col,
                        title=f"Distribution of {value_col} by {categorical_cols[0]}"
                    )
                    # Add hover info to verify values are being used correctly
                    fig.update_traces(
                        hovertemplate="<b>%{label}</b><br>" +
                                    f"{value_col}: %{{value}}<br>" +
                                    "Percentage: %{percent}<br>" +
                                    "<extra></extra>"
                    )
                    debug_info["pie_chart_created"] = True
                else:
                    st.error("No valid numeric data for pie chart after conversion attempts")
                    return None
            else:
                # Only count occurrences if no numeric column exists
                debug_info["using_counts"] = True
                counts = df[categorical_cols[0]].value_counts()
                fig = px.pie(values=counts.values, names=counts.index,
                           title=f"Count Distribution of {categorical_cols[0]}")
                debug_info["pie_chart_created"] = True
                
        elif chart_type == "histogram" and numeric_cols:
            fig = px.histogram(df, x=numeric_cols[0],
                             title=f"Distribution of {numeric_cols[0]}")
            
        elif chart_type == "box" and categorical_cols and numeric_cols:
            fig = px.box(df, x=categorical_cols[0], y=numeric_cols[0],
                        title=f"{numeric_cols[0]} distribution by {categorical_cols[0]}")
            
        else:
            # Fallback: try to create a chart with any available columns
            if len(df.columns) >= 2:
                # Use first column as X and second as Y
                first_col = df.columns[0]
                second_col = df.columns[1]
                
                # Try to create a bar chart
                try:
                    fig = px.bar(df, x=first_col, y=second_col,
                               title=f"{second_col} by {first_col}")
                except:
                    # If that fails, try scatter
                    try:
                        fig = px.scatter(df, x=first_col, y=second_col,
                                       title=f"{second_col} vs {first_col}")
                    except:
                        return None
            elif numeric_cols:
                # Single numeric column - create histogram
                fig = px.histogram(df, x=numeric_cols[0],
                                 title=f"Distribution of {numeric_cols[0]}")
            else:
                return None
                
        fig.update_layout(height=500, showlegend=True)
        
        # Display unified debug information at the bottom (only for pie charts)
        if chart_type == "pie" and 'debug_info' in locals():
            with st.expander("🔧 Visualization Debug", expanded=False):
                st.write(f"🔍 Debug - Original numeric columns: {debug_info['original_numeric_cols']}")
                st.write(f"🔍 Debug - Converted object columns to numeric: {debug_info['converted_cols']}")
                st.write(f"🔍 Debug - All numeric columns: {debug_info['all_numeric_cols']}")
                st.write(f"🔍 Debug - Detected numeric_cols: {debug_info['detected_numeric_cols']}")
                st.write(f"🔍 Debug - Selected value_col: {debug_info['selected_value_col']}")
                
                if debug_info['using_counts']:
                    st.write("🔍 Debug - No numeric column found, using counts")
                elif debug_info.get('pie_data') is not None:
                    st.write(f"🔍 Debug - Using '{debug_info['names_col']}' as names and '{debug_info['values_col']}' as values")
                    st.write("🔍 Debug - Actual pie chart data:")
                    st.dataframe(debug_info['pie_data'])
                    st.write(f"🔍 Debug - Value column '{debug_info['values_col']}' values: {debug_info['value_column_values']}")
                    st.write(f"🔍 Debug - Value column dtype: {debug_info['value_column_dtype']}")
                    st.write(f"🔍 Debug - Name column '{debug_info['names_col']}' values: {debug_info['name_column_values']}")
                
                st.write(f"🔍 Debug - Pie chart created successfully: {debug_info['pie_chart_created']}")
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def suggest_visualization_insights(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    """Generate insights about the data for visualization context"""
    insights = []
    
    insights.append(f"📊 Dataset contains {analysis['row_count']} rows and {analysis['column_count']} columns.")
    
    numeric_cols = [col for col, info in analysis["columns"].items() if info["is_numeric"]]
    categorical_cols = [col for col, info in analysis["columns"].items() if info["is_categorical"]]
    
    if numeric_cols:
        insights.append(f"📈 Numeric columns: {', '.join(numeric_cols)}")
    if categorical_cols:
        insights.append(f"🏷️ Categorical columns: {', '.join(categorical_cols)}")
        
    if analysis["suggested_charts"]:
        chart_names = {
            "bar": "Bar Chart", "scatter": "Scatter Plot", "line": "Line Chart",
            "pie": "Pie Chart", "histogram": "Histogram", "box": "Box Plot"
        }
        suggested = [chart_names[chart] for chart in analysis["suggested_charts"] if chart in chart_names]
        insights.append(f"💡 Recommended visualizations: {', '.join(suggested)}")
    
    return "\n\n".join(insights)

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

# Query Input - Using form to enable Enter key submission
with st.form("query_form", clear_on_submit=False):
    query = st.text_input("Ask a question about your dbt models or request a SQL query:", 
                         "Which models have a foreign key to customers?")
    ask_clicked = st.form_submit_button("Ask")

# Process Query
if ask_clicked:
    with st.spinner("Thinking..."):
        try:
            # Clear any previous visualization state when new query is executed
            viz_keys_to_clear = [key for key in st.session_state.keys() if isinstance(key, str) and key.startswith("viz_analysis_")]
            for key in viz_keys_to_clear:
                del st.session_state[key]
                
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
                
                # Download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "query_results.csv",
                        "text/csv",
                        key=f"download_csv_{hash(st.session_state.current_query)}"
                    )
                
                # Visualization Analysis and Creation
                with col2:
                    viz_clicked = st.button("📈 Create Visualization", 
                                          key=f"create_viz_{hash(st.session_state.current_query)}")
                
                if viz_clicked or f"viz_analysis_{hash(st.session_state.current_query)}" in st.session_state:
                    # Store analysis in session state to persist across reruns
                    analysis_key = f"viz_analysis_{hash(st.session_state.current_query)}"
                    if analysis_key not in st.session_state:
                        st.session_state[analysis_key] = analyze_dataframe_for_viz(df)
                    
                    analysis = st.session_state[analysis_key]
                    
                    if analysis["can_visualize"] and analysis["suggested_charts"]:
                        st.markdown("### 📈 Data Visualization")
                        
                        # Show insights about the data
                        with st.expander("📋 Data Analysis Insights", expanded=False):
                            insights = suggest_visualization_insights(df, analysis)
                            st.markdown(insights)
                            
                            # Debug information to help troubleshoot visualization issues
                            st.markdown("**Column Analysis:**")
                            for col, info in analysis["columns"].items():
                                st.markdown(f"- **{col}**: {info['dtype']} | Unique: {info['unique_count']} | Numeric: {info['is_numeric']} | Categorical: {info['is_categorical']}")
                            
                            # Show column lists for debugging
                            numeric_cols = [col for col, info in analysis["columns"].items() if info["is_numeric"]]
                            categorical_cols = [col for col, info in analysis["columns"].items() if info["is_categorical"]]
                            st.markdown(f"**Detected Numeric Columns**: {numeric_cols}")
                            st.markdown(f"**Detected Categorical Columns**: {categorical_cols}")
                            
                            # Show sample of actual data
                            st.markdown("**Sample Data:**")
                            st.dataframe(df.head(3), use_container_width=True)
                            
                            # Show data types from pandas
                            st.markdown("**Pandas Data Types:**")
                            st.code(str(df.dtypes))
                        
                        # Chart selection
                        chart_options = {
                            "bar": "📊 Bar Chart",
                            "scatter": "🔵 Scatter Plot", 
                            "line": "📈 Line Chart",
                            "pie": "🥧 Pie Chart",
                            "histogram": "📊 Histogram",
                            "box": "📦 Box Plot"
                        }
                        
                        available_charts = [chart for chart in analysis["suggested_charts"] if chart in chart_options]
                        
                        if available_charts:
                            selected_chart = st.selectbox(
                                "Choose visualization type:",
                                available_charts,
                                format_func=lambda x: chart_options.get(x, str(x)),
                                key=f"chart_select_{hash(st.session_state.current_query)}"
                            )
                            
                            # Create and display the visualization
                            if selected_chart:
                                with st.spinner("Creating visualization..."):
                                    fig = create_visualization(df, selected_chart, analysis)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Could not create the selected visualization for this data.")
                        else:
                            st.info("No suitable visualizations available for this dataset.")
                    else:
                        if not analysis["can_visualize"]:
                            st.info(f"Visualization not available: {analysis.get('reason', 'Unknown reason')}")
                        else:
                            st.info("No suitable visualization patterns detected in this dataset.")
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
