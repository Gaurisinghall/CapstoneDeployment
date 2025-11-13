import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import your existing query pipeline here
from app.retrieval import run, load_history, append_history, HISTORY_FILE 
from app.ingestion import ingest_all_pdfs 


class QueryRequest(BaseModel):
    query: str


app = FastAPI(
    title="Medical Device Documentation System",
    description="AI-powered medical device documentation system",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Mount static and templates relative to package
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)



@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query")
async def api_query(request: QueryRequest):
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Await run() because it's an async coroutine
    answer = await run(q)  
    return {"query": q, "answer": answer}


@app.get("/history")
async def get_history_api(limit: int = 20):
    try:
        history = load_history(n=limit)
    except Exception:
        history = []
    return {"history": history[::-1]}  # newest entries first


@app.delete("/history")
async def clear_history_api():
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return {"message": "History cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear history: " + str(e))
    
@app.get("/ingest")
def ingest():
    try:
        ingest_all_pdfs()
        return {"status": "ingestion complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
