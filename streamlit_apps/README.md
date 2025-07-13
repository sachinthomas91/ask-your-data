# DBT Model Query Interface

This Streamlit application provides a natural language interface to query your DBT models. It uses ChromaDB for storing model documentation embeddings and Ollama for both embeddings and text generation.

## Features

- Natural language queries about your DBT models
- Automatic SQL query generation
- Model documentation search
- Interactive model exploration

## Requirements

- Python 3.8+
- Streamlit
- ChromaDB
- Ollama
- Required Ollama models:
  - `nomic-embed-text` for embeddings
  - `mistral` for text generation (you can configure a different model)

## Usage

1. First, run the embedding script to create the vector store:
   ```bash
   python src/semantic_models/embed_dbt_models.py
   ```

2. Then start the Streamlit app:
   ```bash
   streamlit run streamlit_apps/app_nl_query.py
   ```

## Screenshot

![Sample Query](../assets/images/sample_query.png)

## How it Works

1. The app uses ChromaDB to store embeddings of your DBT model documentation
2. When you ask a question, it:
   - Finds relevant model documentation using semantic search
   - Uses Ollama to generate a natural language answer or SQL query
   - Shows you the referenced models for transparency

## Configuration

You can configure different models by changing these constants in `app_nl_query.py`:
```python
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "mistral"  # or any other model you prefer
```
