Backend - FastAPI (SQLite)
==========================

Run the backend (from project root):

1. Create & activate virtualenv (recommended):
   python -m venv venv
   source venv/bin/activate   (Windows: venv\Scripts\activate)

2. Install requirements:
   pip install -r backend/requirements.txt

3. Start the app (run from project root):
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

API endpoints:
 - GET  /items/       -> list items
 - POST /items/       -> create item (JSON body: {"name":..., "description":...})
 - DELETE /items/{id} -> delete item
