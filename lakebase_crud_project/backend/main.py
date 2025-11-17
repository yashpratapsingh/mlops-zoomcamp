from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from databricks import sql

# Simple in-memory fallback if Databricks config not provided (for local dev/testing)
USE_DATABRICKS = os.getenv("USE_DATABRICKS", "false").lower() == "true"

app = FastAPI(title="Lakebase CRUD API")

class Customer(BaseModel):
    id: int | None = None
    name: str
    email: str
    city: str

# In-memory store (id auto-increment)
_store = {}
_next_id = 1

# Databricks connection helper (lazy)
_conn = None
def get_db_conn():
    global _conn
    if _conn is not None:
        return _conn
    if not USE_DATABRICKS:
        return None
    # Expect env vars: DATABRICKS_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
    try:
        _conn = sql.connect(
            server_hostname=os.environ["DATABRICKS_HOSTNAME"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"]
        )
        return _conn
    except Exception as e:
        print('Failed to connect to Databricks SQL:', e)
        return None

@app.on_event("startup")
def startup():
    # Initialize store with sample data for quick testing
    global _store, _next_id
    _store = {
        1: {"id":1, "name":"Alice", "email":"alice@example.com", "city":"Delhi"},
        2: {"id":2, "name":"Bob", "email":"bob@example.com", "city":"Mumbai"}
    }
    _next_id = 3

@app.get("/customers", response_model=List[Customer])
def list_customers():
    conn = get_db_conn()
    if conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, email, city FROM lakebase_poc.customers")
        rows = cur.fetchall()
        return [{"id":r[0], "name":r[1], "email":r[2], "city":r[3]} for r in rows]
    # fallback
    return list(_store.values())

@app.post("/customers", response_model=Customer)
def create_customer(c: Customer):
    conn = get_db_conn()
    global _next_id
    if conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO lakebase_poc.customers (name, email, city) VALUES (%s, %s, %s) RETURNING id",
            (c.name, c.email, c.city)
        )
        new_id = cur.fetchone()[0]
        c.id = new_id
        return c
    # fallback
    c.id = _next_id
    _store[_next_id] = c.dict()
    _next_id += 1
    return c

@app.put("/customers/{customer_id}", response_model=Customer)
def update_customer(customer_id: int, c: Customer):
    conn = get_db_conn()
    if conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE lakebase_poc.customers SET name=%s, email=%s, city=%s WHERE id=%s",
            (c.name, c.email, c.city, customer_id)
        )
        return {"id": customer_id, "name": c.name, "email": c.email, "city": c.city}
    # fallback
    if customer_id not in _store:
        raise HTTPException(status_code=404, detail="Customer not found")
    c.id = customer_id
    _store[customer_id] = c.dict()
    return c

@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: int):
    conn = get_db_conn()
    if conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM lakebase_poc.customers WHERE id=%s", (customer_id,))
        return {"status":"deleted"}
    # fallback
    if customer_id not in _store:
        raise HTTPException(status_code=404, detail="Customer not found")
    del _store[customer_id]
    return {"status":"deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
