import os, re, json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from crewai import Agent, Task, Crew, LLM
from openai import AzureOpenAI

# ======================================
# Env + clients
# ======================================
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

BASE_DATA_DIR = os.getenv("APP_DATA_DIR", "/home")
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DATA_DIR, "data/chroma"))
os.makedirs(CHROMA_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

azure_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_type="azure",
    api_version=AZURE_API_VERSION,
    deployment_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)
collection = chroma_client.get_or_create_collection(
    name="manual_chunks",
    embedding_function=azure_ef,
    metadata={"hnsw:space": "cosine"},
)

# ======================================
# Product mapping + normalization
# ======================================
PRODUCT_TO_PDF = {
    "vs8": "vs8_vital_signs_monitor.pdf",
    "dp50": "dp50_ultrasound_procedures.pdf",
    "mx40": "mx40_wearable_monitor.pdf",
    "pic ix": "pic_ix_patient_info_centre.pdf",
}

PRODUCT_PATTERNS = [
    (r"\bvs[- ]?8\b", "vs8"),
    (r"\bdp[- ]?50\b", "dp50"),
    (r"\bmx[- ]?40\b", "mx40"),
    (r"\bpic[- ]?ix\b", "pic ix"),
]

def normalize_query(text: str) -> str:
    q = text.lower().strip()
    replacements = {
        r"\bpower of\b": "power off",
        r"\bshut down\b": "power off",
        r"\btemp\b": "temperature",
        r"\bbatt(ery)?\b": "battery",
        r"\bspecs\b": "specifications",
        r"\bdimension(s)?\b": "dimensions",
    }
    for pattern, replacement in replacements.items():
        q = re.sub(pattern, replacement, q)
    q = re.sub(r"\s+", " ", q)
    return q

def match_products(q: str) -> List[str]:
    hits = set()
    for pat, key in PRODUCT_PATTERNS:
        if re.search(pat, q):
            hits.add(key)
    for key in PRODUCT_TO_PDF.keys():
        if key in q:
            hits.add(key)
    return list(hits)

# ======================================
# Query engine + retrieval
# ======================================
class QueryEngine:
    def __init__(self, n_results=10):
        self.n_results = n_results
        self.collection = collection

    def run(self, query: str, product_pdf: Optional[str]) -> List[Dict[str, Any]]:
        filters = {"source_pdf": {"$eq": product_pdf}} if product_pdf else None
        res = self.collection.query(
            query_texts=[query],
            n_results=self.n_results,
            where=filters,
        )
        if not res or not res.get("documents"):
            return []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        out = []
        for doc, meta in zip(docs, metas):
            # Guard: ensure every passage matches product_pdf
            if product_pdf and meta.get("source_pdf") != product_pdf:
                continue
            out.append({
                "doc": doc,
                "source_pdf": meta.get("source_pdf"),
                "page_number": meta.get("page_number"),
                "type": meta.get("type"),
                "section_heading": meta.get("section_heading"),
            })
        return out

def rerank_unique(passages: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for p in passages:
        key = (p.get("source_pdf"), p.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= top_n:
            break
    return out

# ======================================
# History utils (scoped)
# ======================================

HISTORY_FILE = os.getenv("HISTORY_FILE", os.path.join(BASE_DATA_DIR, "conversation_history.json"))
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
MAX_HISTORY = 500

def load_history(n: int = 3, product_pdf: str = None) -> List[Dict[str, Any]]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            if product_pdf:
                history = [h for h in history if h.get("product") == product_pdf]
            return history[-n:]
    except Exception:
        pass
    return []

def append_history(entry: Dict[str, Any]) -> None:
    try:
        all_hist = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                all_hist = json.load(f)
        all_hist.append(entry)
        if len(all_hist) > MAX_HISTORY:
            all_hist = all_hist[-MAX_HISTORY:]
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_hist, f, ensure_ascii=False, indent=2)
        os.replace(tmp, HISTORY_FILE)
    except Exception:
        pass

# ======================================
# Synthesis (strictly from context, scoped history)
# ======================================
def synthesize(query: str, product_pdf: Optional[str], passages: List[Dict[str, Any]], history_n: int = 3) -> str:
    # Guard: ensure passages belong to the intended product
    if product_pdf:
        passages = [p for p in passages if p.get("source_pdf") == product_pdf]
    if not passages:
        return f"No relevant information found for {product_pdf or 'this product'}."

    context = "\n\n".join(
        f"{p['doc']} [{p['source_pdf']}, p. {p.get('page_number', '?')}]"
        for p in passages
    )
    history_entries = load_history(history_n, product_pdf=product_pdf)
    history_context = "\n".join(
        f"Previous Q: {h['query']}\nPrevious A: {h['answer']}" for h in history_entries
    )
    system_prompt = (
        "You are a helpful assistant for medical device manuals.\n"
        "Answer strictly using the provided context.\n"
        "After every factual claim, include an inline citation like [DocTitle.pdf, p. X].\n"
        "If the information is not present, say it cannot be found in the available documents."
    )
    user_prompt = (
        f"Conversation history:\n{history_context}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer specifically for {product_pdf or 'the specified product'}, with citations and continuity."
    )
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content
    except Exception:
        return "\n\n".join(
            f"{p['doc']} [{p['source_pdf']}, p. {p.get('page_number', '?')}]"
            for p in passages
        )

# ======================================
# Agents
# ======================================
llm_config = LLM(
    model=f"azure/{AZURE_OPENAI_CHAT_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
    temperature=0.3
)

query_parser = Agent(name="QueryParser", role="Parse queries", goal="Extract product names and map to manuals", backstory="Targets correct manuals.", llm=llm_config)
retriever = Agent(name="Retriever", role="Retrieve chunks", goal="Run Chroma similarity search with strict filters", backstory="Ensures page-level traceability.", llm=llm_config)
synthesizer = Agent(name="Synthesizer", role="Generate answers", goal="Ground responses in retrieved passages with citations", backstory="No hallucinations.", llm=llm_config)
history_manager = Agent(name="HistoryManager", role="Maintain dialogue context", goal="Store/retrieve scoped history", backstory="Keeps product-specific continuity.", llm=llm_config)
context_integrator = Agent(name="ContextIntegrator", role="Merge answers with history", goal="Combine per-product answers cleanly", backstory="Final coherent output.", llm=llm_config)

# ======================================
# Pipeline (stepwise, one agent at a time)
# ======================================
qe = QueryEngine(n_results=10)
def parse_query(text: str, fallback_n: int = 2) -> Dict[str, Any]:
    """
    Parse user query into per-product sub_queries.
    If no product is detected, fall back to the last `fallback_n` products
    referenced in history (most recent first).
    Returns: {"sub_queries": [{"raw": text, "product_pdf": <pdf or None>}, ...]}
    """
    q = normalize_query(text)
    keys = match_products(q)  # returns canonical keys like "mx40", "dp50"
    if keys:
        subs = [{"raw": text, "product_pdf": PRODUCT_TO_PDF[k]} for k in keys]
        return {"sub_queries": subs}

    # No explicit product mentioned — attempt fallback from history
    recent = load_history(50)  # get a window of recent interactions
    recent_products = []
    for h in reversed(recent):  # iterate most recent first
        p = h.get("product")
        if p and p not in recent_products:
            recent_products.append(p)
        if len(recent_products) >= fallback_n:
            break

    if recent_products:
        # Use the most recent product(s) as fallback, but mark that they were inferred
        subs = [{"raw": text, "product_pdf": p, "inferred_from_history": True} for p in recent_products]
        return {"sub_queries": subs}

    # No history fallback available — return unspecified
    return {"sub_queries": [{"raw": text, "product_pdf": None}]}

def run_retrieval(sub_query: dict) -> List[Dict[str, Any]]:
    passages = qe.run(sub_query["raw"], product_pdf=sub_query.get("product_pdf"))
    return rerank_unique(passages, top_n=5)

def run_synthesis(query: str, product_pdf: Optional[str], passages: List[Dict[str, Any]]) -> str:
    return synthesize(query, product_pdf, passages)

def integrate_context(query: str, product_answers: List[Dict[str, Any]], history_n: int = 2) -> str:
    """
    Merge per-product answers while preserving concise, product-scoped history.
    If all product_answers are unspecified, return a clarification prompt.
    """
    # If user didn't specify any product and nothing was inferred, ask for clarification
    if all(not pa.get("product") for pa in product_answers):
        return (
            "I couldn't detect a specific product in your question. "
            "Please specify which product you mean (e.g., DP50, MX40, VS8) so I can provide focused information."
        )

    merged_blocks = []
    for pa in product_answers:
        product_pdf = pa.get("product")
        inferred = pa.get("inferred_from_history", False)

        # Scoped recent history for this product only
        scoped_hist = load_history(history_n, product_pdf=product_pdf) if product_pdf else []
        hist_block = ""
        if scoped_hist:
            lines = []
            for h in scoped_hist:
                q = h.get("query", "").strip()
                a = h.get("answer", "").strip()
                q_short = (q[:120] + "...") if len(q) > 120 else q
                a_short = (a[:200] + "...") if len(a) > 200 else a
                lines.append(f"- Q: {q_short} → A: {a_short}")
            hist_block = "Recent (product):\n" + "\n".join(lines) + "\n\n"

        title = (product_pdf or "unspecified product").replace("_", " ").replace(".pdf", "")
        inferred_note = " (inferred from recent queries)" if inferred else ""
        merged_blocks.append(f"**{title}**{inferred_note}:\n{hist_block}{pa['answer']}")

    return "\n\n".join(merged_blocks)


# ======================================
# Crew tasks
# ======================================
parse_task = Task(agent=query_parser, description="Parse query into sub-queries.", expected_output="Dict with sub_queries list.", function=parse_query)
retrieval_task = Task(agent=retriever, description="Retrieve and rerank per sub-query.", expected_output="List of passages.", function=run_retrieval)
synthesis_task = Task(agent=synthesizer, description="Generate grounded answer per product.", expected_output="Answer string.", function=lambda args: run_synthesis(args["query"], args.get("product_pdf"), args.get("passages", [])))
context_task = Task(agent=context_integrator, description="Merge per-product answers with history.", expected_output="Final coherent answer.", function=lambda args: integrate_context(args["query"], args["product_answers"]))
history_task = Task(agent=history_manager, description="Persist outputs.", expected_output="Updated history file.", function=lambda args: append_history({"query": args["query"], "answer": args["final_answer"], "product": args.get("product")}))

crew = Crew(
    agents=[query_parser, retriever, synthesizer, context_integrator, history_manager],
    tasks=[parse_task, retrieval_task, synthesis_task, context_task, history_task],
    verbose=True
)

# ======================================
# Deterministic orchestration
# ======================================
def run_with_crew(query: str) -> Dict[str, Any]:
    plan = parse_query(query)
    product_answers = []
    for sq in plan["sub_queries"]:
        passages = run_retrieval(sq)
        answer = run_synthesis(sq["raw"], sq.get("product_pdf"), passages)
        product_answers.append({"product": sq.get("product_pdf") or "unspecified product", "answer": answer})
        append_history({"query": sq["raw"], "answer": answer, "product": sq.get("product_pdf")})
    final_answer = integrate_context(query, product_answers)
    append_history({"query": query, "answer": final_answer, "product": None})
    return {"answer": final_answer}

def answer_query(query: str) -> Dict[str, Any]:
    """
    Deployment-ready entry point.
    Runs the same pipeline as the CLI:
      - normalize query
      - parse into sub-queries (explicit products or inferred from history)
      - deterministic retrieval (no LLM)
      - synthesis with Azure LLM
      - integrate context
      - update history
    Returns a dict with the normalized query, final answer, and per-product details.
    """
    normalized = normalize_query(query)

    # Step 1: parse into sub-queries
    plan = parse_query(normalized)
    subs = plan["sub_queries"]

    # Step 2: handle follow-up if no explicit product
    if subs and subs[0].get("product_pdf") is None:
        # fallback to recent history if available
        recent = load_history(50)
        if recent:
            recent_products = []
            for h in reversed(recent):
                p = h.get("product")
                if p and p not in recent_products:
                    recent_products.append(p)
                if len(recent_products) >= 2:
                    break
            if recent_products:
                subs = [{"raw": normalized, "product_pdf": p, "inferred_from_history": True} for p in recent_products]

    product_answers = []

    # Step 3: run each subquery deterministically (retrieval + synthesis)
    for sub in subs:
        passages = run_retrieval(sub)
        answer = run_synthesis(sub["raw"], sub.get("product_pdf"), passages)
        pa = {
            "product": sub.get("product_pdf"),
            "answer": answer,
            "query": sub["raw"],
            "inferred_from_history": sub.get("inferred_from_history", False)
        }
        product_answers.append(pa)
        append_history({"query": pa["query"], "answer": pa["answer"], "product": pa["product"]})

    # Step 4: integrate context into final answer
    final_answer = integrate_context(normalized, product_answers)
    append_history({"query": normalized, "answer": final_answer, "product": None})

    # Step 5: return structured result
    return {
        "query": normalized,
        "answer": final_answer,
        "details": product_answers
    }

# ======================================
# Disable CLI for deployment
# ======================================
if __name__ == "__main__":
    # For local testing only
    print("⚠️ CLI mode is disabled in deployment. Use answer_query() via FastAPI or app entrypoint.")
