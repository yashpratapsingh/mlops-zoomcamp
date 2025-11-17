Backend (FastAPI)
-----------------
- Run locally:
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

- To use Databricks SQL Warehouse:
    1. Create a SQL Warehouse (Serverless/Classic) and copy hostname, http path and token.
    2. Set environment variables or use a .env loader:
        USE_DATABRICKS=true
        DATABRICKS_HOSTNAME=...
        DATABRICKS_HTTP_PATH=...
        DATABRICKS_TOKEN=...
