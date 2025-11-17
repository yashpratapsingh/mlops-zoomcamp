# Lakebase CRUD Project (React + FastAPI)

## What you get
- Backend: FastAPI (backend/)
- Frontend: React + Vite + Tailwind (frontend/)
- Dockerfile for backend
- Example .env and instructions

## Quick start (local)
1. Backend:
    cd backend
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

2. Frontend:
    cd frontend
    npm install
    npm run dev

## Using Databricks Lakebase
- Set USE_DATABRICKS=true and provide DATABRICKS_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN environment variables.
- The backend uses `databricks-sql-connector` to run SQL against Lakebase. The code currently assumes a schema/table `lakebase_poc.customers`.
- Create the table in your Lakebase (Databricks) SQL endpoint:
    CREATE TABLE lakebase_poc.customers (
      id INT GENERATED ALWAYS AS IDENTITY,
      name STRING,
      email STRING,
      city STRING
    );

## Docker
- Build backend image:
    docker build -t lakebase-backend ./backend
- Run:
    docker run -e USE_DATABRICKS=false -p 8000:8000 lakebase-backend

## Notes
- For production, add proper secrets management, CORS, input validation, and use parameterized queries (the databricks connector supports parameterization).
