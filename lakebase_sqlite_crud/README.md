Lakebase SQLite CRUD (FastAPI + Vite React)
==========================================

How to run locally (tested in Codespaces / local machine)

1) Backend
   cd backend
   python -m venv venv
   source venv/bin/activate   (Windows: venv\Scripts\activate)
   pip install -r backend/requirements.txt
   cd ..
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

2) Frontend (new terminal)
   cd frontend
   npm install
   npm run dev
   Open http://localhost:5173

Notes:
- Backend uses SQLite file: backend/lakebase.db
- To switch to Postgres/Databricks later, update backend/database.py connection string.
