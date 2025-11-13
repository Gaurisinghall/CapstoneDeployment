import warnings
import logging
import os
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import AzureOpenAI

from crewai.flow.flow import Flow, start, listen
from crewai import Agent, Task, Crew, LLM, Process


# Suppress CrewAI Flow verbose output
warnings.filterwarnings("ignore")
logging.getLogger("crewai").setLevel(logging.ERROR)
logging.getLogger("crewai.flow").setLevel(logging.ERROR)

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

llm_azure = LLM(
    model=f"azure/{AZURE_OPENAI_CHAT_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION,
    temperature=0.2,
)
BASE_DATA_DIR = os.getenv("APP_DATA_DIR", "/home")
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DATA_DIR, "data/chroma"))
os.makedirs(CHROMA_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# chroma_client = chromadb.PersistentClient(path="data/chroma")

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

HISTORY_FILE = os.getenv("HISTORY_FILE", os.path.join(BASE_DATA_DIR, "conversation_history.json"))
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
# HISTORY_FILE = "conversation_history.json"
MAX_HISTORY = 500

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
        entry["timestamp"] = datetime.now().isoformat()
        all_hist.append(entry)
        if len(all_hist) > MAX_HISTORY:
            all_hist = all_hist[-MAX_HISTORY:]
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_hist, f, ensure_ascii=False, indent=2)
        os.replace(tmp, HISTORY_FILE)
    except Exception:
        pass


def get_recent_products(n: int = 2) -> List[str]:
    """Get recently discussed products from history."""
    recent = load_history(50)
    recent_products = []
    for h in reversed(recent):
        p = h.get("product")
        if p and p not in recent_products:
            recent_products.append(p)
        if len(recent_products) >= n:
            break
    return recent_products


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


def synthesize(query: str, product_pdf: Optional[str], passages: List[Dict[str, Any]], history_n: int = 3) -> str:
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
        f"Answer with citations."
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
    except Exception as e:
        print(f"Synthesis error: {e}")
        return "\n\n".join(
            f"{p['doc']} [{p['source_pdf']}, p. {p.get('page_number', '?')}]"
            for p in passages
        )


qe = QueryEngine(n_results=10)


query_parser_agent = Agent(
    name="QueryParser",
    role="Parse queries into product-specific sub-queries",
    goal="Split multi-product questions into focused sub-queries",
    backstory="Expert at understanding medical device queries",
    llm=llm_azure,
    verbose=False,
)


integration_agent = Agent(
    name="ContextIntegrator",
    role="Merge answers into coherent final response",
    goal="Combine product-specific answers maintaining context",
    backstory="Skilled at synthesizing information clearly",
    llm=llm_azure,
    verbose=False,
)


class DocumentRetrievalFlow(Flow):
    """Flow-based orchestration matching CrewAI 1.4.1 patterns."""
    
    @start()
    def initialize_query(self):
        
        user_query = self.state.get("user_query", "")
        normalized_query = normalize_query(user_query)
        
        detected_products = match_products(normalized_query)

        if not detected_products:
            recent = get_recent_products(2)
            if recent:

                sub_queries = [
                    {"product": p, "question": user_query, "inferred_from_history": True}
                    for p in recent
                ]
            else:

                sub_queries = [{"product": None, "question": user_query}]
        else:
            sub_queries = [
                {"product": PRODUCT_TO_PDF[p], "question": user_query}
                for p in detected_products
            ]
        
        # print(f"Sub-queries: {len(sub_queries)}")
        
        # Store in state
        self.state["normalized_query"] = normalized_query
        self.state["sub_queries"] = sub_queries
        
        return sub_queries
    
    @listen(initialize_query)
    def retrieve_documents(self, sub_queries):

        
        product_answers = []
        user_query = self.state.get("user_query", "")
        
        for i, sq in enumerate(sub_queries, 1):
            product = sq.get("product")
            question = sq.get("question", user_query)
            
            
            passages = qe.run(question, product_pdf=product)
            passages = rerank_unique(passages, top_n=5)
            
            # Synthesize answer
            answer = synthesize(question, product, passages, history_n=3)
            
            product_answers.append({
                "product": product or "unspecified",
                "answer": answer,
                "inferred_from_history": sq.get("inferred_from_history", False)
            })
            
            # Save to history
            append_history({
                "query": question,
                "answer": answer,
                "product": product,
            })
        
        self.state["product_answers"] = product_answers
        return product_answers
    
    @listen(retrieve_documents)
    def integrate_answers(self, product_answers):

        
        user_query = self.state.get("user_query", "")
        
        if not product_answers:
            final_answer = "No results found."
        else:

            answers_summary = "\n\n".join(
                f"**{pa['product']}**:\n{pa['answer']}"
                for pa in product_answers
            )
            
            try:

                task = Task(
                    agent=integration_agent,
                    description=(
                        f"Integrate these answers into ONE coherent final response.\n"
                        f"Original Query: '{user_query}'\n\n"
                        f"Retrieved Information:\n{answers_summary}\n\n"
                        f"Please provide a comprehensive answer based on the retrieved information."
                    ),
                    expected_output="Final integrated answer with citations",
                )
                crew = Crew(
                    agents=[integration_agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=False,
                )
                result = crew.kickoff({"answers": answers_summary})
                final_answer = str(result)
                
                # print(f"✅ CrewAI Integration Complete")
                
            except Exception as e:

                final_answer = answers_summary
        
        # Save final answer
        append_history({
            "query": user_query,
            "answer": final_answer,
            "product": None,
        })
        
        self.state["final_answer"] = final_answer
        return final_answer


async def run(query: str) -> str:
    flow = DocumentRetrievalFlow()
    flow.state["user_query"] = query
    result = await flow.kickoff_async() 
    return result


