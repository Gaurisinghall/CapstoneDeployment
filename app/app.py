import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import retrieval + ingestion functions
from app.retrieval import answer_query, load_history, HISTORY_FILE
from app.ingestion import ingest_all_pdfs

app = FastAPI()

# ---------------------------
# Resolve base directory
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static and templates relative to package
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

# ---------------------------
# Backend API endpoints
# ---------------------------
@app.get("/rag")
def rag(query: str):
    return answer_query(query)

@app.get("/history")
def get_history():
    return {"history": load_history(10)}

@app.get("/clear-history")
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        return {"status": "cleared"}
    return {"status": "no history found"}

@app.get("/ingest")
def ingest():
    ingest_all_pdfs()
    return {"status": "ingestion complete"}

# ---------------------------
# Frontend route
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request, query: str = "", action: str = ""):
    answer = ""
    details = []
    history = []
    status = ""

    if query:
        data = answer_query(query)
        answer = data.get("answer", "")
        details = data.get("details", [])

    if action == "history":
        history = load_history(10)

    if action == "clear":
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            status = "cleared"
        else:
            status = "no history found"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": query,
            "answer": answer,
            "details": details,
            "history": history,
            "status": status,
        },
    )

