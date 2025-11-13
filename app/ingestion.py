import os
import re
import json
import time
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

import fitz
import pdfplumber
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-capstone")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large-capstone")

BASE_DATA_DIR = os.getenv("APP_DATA_DIR", "/home")

CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DATA_DIR, "data/chroma"))
os.makedirs(CHROMA_PATH, exist_ok=True)

HISTORY_FILE = os.getenv("HISTORY_FILE", os.path.join(BASE_DATA_DIR, "conversation_history.json"))
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


blob_service_client = BlobServiceClient(
    account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
    credential=AZURE_STORAGE_ACCOUNT_KEY
)
container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER)

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

azure_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_type="azure",
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "manual_chunks")
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=azure_embedding_fn,
    metadata={"hnsw:space": "cosine"}
)


class BlobLoader:
    def list_blobs(self) -> List[str]:
        return [blob.name for blob in container_client.list_blobs()]

    def download_blob_as_bytes(self, blob_name: str) -> bytes:
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()


def extract_text_and_tables_from_blob(blob_bytes: bytes) -> List[Any]:
    """
    Extract text (via PyMuPDF) and tables (via pdfplumber) from PDF bytes.
    Returns a unified list of dicts with type, content, and page_number.
    """
    pdf_stream = BytesIO(blob_bytes)

    elements = []

    # --- Text extraction with PyMuPDF ---
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")  # layout-aware text blocks
        for b in blocks:
            text = b[4].strip()
            if text:
                elements.append({
                    "type": "text",
                    "content": text,
                    "page_number": page_num
                })

    # Reset stream for pdfplumber
    pdf_stream.seek(0)

    # --- Table extraction with pdfplumber ---
    with pdfplumber.open(pdf_stream) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                # table is a list of rows, each row is a list of cell strings
                elements.append({
                    "type": "table",
                    "content": table,
                    "page_number": page_num
                })

    return elements

def extract_images_from_pdf(blob_bytes: bytes, source_pdf: str) -> List[Dict[str, Any]]:
    """
    Extract images per page with PyMuPDF, include page_number and a stable chunk_id.
    """
    images: List[Dict[str, Any]] = []
    doc = fitz.open(stream=blob_bytes, filetype="pdf")
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width = base_image.get("width")
            height = base_image.get("height")
            images.append({
                "chunk_id": f"{source_pdf}_page{page_idx+1}_img{img_index+1}",
                "image_bytes": image_bytes,
                "source_pdf": source_pdf,
                "page_number": page_idx + 1,
                "type": "image",
                "width": width,
                "height": height
            })
    return images

def get_checkpoint_file(pdf_name: str, prefix: str = "captions") -> str:
    """
    Generate a per-PDF checkpoint path under /home (Azure writable directory).
    """
    base = os.path.splitext(os.path.basename(pdf_name))[0]
    return os.path.join(BASE_DATA_DIR, f"{prefix}_{base}.json")



def safe_json_parse(raw: str) -> Optional[Any]:
    """Try to parse JSON, fallback to extracting first array/dict."""
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def repair_json(raw: str) -> Optional[Any]:
    """Ask the model to repair malformed JSON into a valid array of {index, caption}."""
    try:
        repair_response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a strict JSON formatter."},
                {"role": "user", "content": (
                    "The following text was supposed to be a JSON array of objects "
                    "with keys 'index' and 'caption', but it's malformed:\n\n"
                    f"{raw}\n\n"
                    "Please return ONLY a valid JSON array of objects like:\n"
                    "[{\"index\": 1, \"caption\": \"...\"}, {\"index\": 2, \"caption\": \"...\"}]"
                )}
            ],
            max_tokens=800,
            temperature=0
        )
        fixed = repair_response.choices[0].message.content
        return safe_json_parse(fixed)
    except Exception as e:
        print(f"⚠️ Repair step failed: {e}")
        return None


def normalize_captions(parsed: Any) -> List[str]:
    """Flatten parsed JSON into a list of caption strings."""
    captions: List[str] = []
    if isinstance(parsed, str):
        captions.append(parsed)
    elif isinstance(parsed, dict):
        if "caption" in parsed:
            captions.append(str(parsed["caption"]))
        for key in ("output", "captions", "images", "responses", "data"):
            if key in parsed and isinstance(parsed[key], list):
                for item in parsed[key]:
                    if isinstance(item, dict) and "caption" in item:
                        captions.append(str(item["caption"]))
    elif isinstance(parsed, list):
        for item in parsed:
            captions.extend(normalize_captions(item))
    return [c.strip() for c in captions if c and str(c).strip()]


def batch_caption_images(
    images: List[Dict[str, Any]],
    batch_size: int = 10,
    checkpoint_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Caption images in batches using Azure OpenAI chat deployment, with retries and atomic checkpoints.
    Uses multimodal message format for efficiency.
    """
    if checkpoint_file is None and images:
        # Default to /home/captions_<pdf>.json
        source_pdf = images[0]["source_pdf"]
        checkpoint_file = get_checkpoint_file(source_pdf, prefix="captions")

    results: List[Dict[str, Any]] = []

    # Resume from checkpoint if exists
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"🔄 Resuming captions from checkpoint with {len(results)} items")

    done_ids = {r["chunk_id"] for r in results}

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        if not batch or all(img["chunk_id"] in done_ids for img in batch):
            continue

        # Build multimodal content
        content = [{
            "type": "text",
            "text": "Describe each image in one sentence. Return ONLY a JSON array of objects [{'index': N, 'caption': '...'}]."
        }]
        for img in batch:
            b64_image = base64.b64encode(img["image_bytes"]).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"}
            })

        # Retry loop
        for attempt in range(3):
            try:
                resp = openai_client.chat.completions.create(
                    model=AZURE_OPENAI_CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are an assistant that describes medical device images clearly."},
                        {"role": "user", "content": content}
                    ],
                    max_tokens=800,
                )
                raw = resp.choices[0].message.content
                parsed = safe_json_parse(raw) or repair_json(raw) or raw

                iterable = parsed if isinstance(parsed, list) else [parsed]
                for idx, cap in enumerate(iterable):
                    img = batch[idx % len(batch)]
                    caps = normalize_captions(cap) or [str(cap)]
                    for c in caps:
                        results.append({
                            "chunk_id": img["chunk_id"],
                            "caption": c.strip(),
                            "source_pdf": img["source_pdf"],
                            "page_number": img["page_number"],
                            "type": "image_caption"
                        })

                # Atomic checkpoint write
                tmp_file = checkpoint_file + ".tmp"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                os.replace(tmp_file, checkpoint_file)

                print(f"✅ Caption batch {i//batch_size} complete; total {len(results)}")
                break
            except Exception as e:
                print(f"⚠️ Caption batch {i//batch_size} failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"❌ Skipping caption batch {i//batch_size} after 3 failed attempts")

    return results


def chunk_elements(
    elements: List[Any],
    source_pdf: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks: List[Dict[str, Any]] = []
    local_id = 0

    for el in elements:
        if isinstance(el, dict):
            raw_content = el.get("content")
            el_type = (el.get("type") or "text").lower()
            page_number = el.get("page_number")
            section_heading = el.get("section_heading")

            # --- Special handling for tables ---
            if el_type == "table" and isinstance(raw_content, list):
                # Replace None with empty string before joining
                text = "\n".join(
                    [" | ".join([cell if cell is not None else "" for cell in row])
                    for row in raw_content if row]
                )
            else:
                text = (raw_content or "").strip()
        else:
            text = str(el).strip()
            el_type = getattr(el, "category", "text")
            if isinstance(el_type, str):
                el_type = el_type.lower()
            meta = getattr(el, "metadata", None)
            page_number = getattr(meta, "page_number", None) if meta else None
            section_heading = getattr(meta, "section", None) if meta else None

        if not text:
            continue

        if el_type in {"text", "uncategorizedtext", "title", "narrativetext"}:
            sub_chunks = splitter.split_text(text)
        else:
            sub_chunks = [text]

        for sub in sub_chunks:
            chunks.append({
                "chunk_id": f"{source_pdf}_{local_id}",
                "source_pdf": source_pdf,
                "section_heading": section_heading,
                "page_number": page_number,
                "type": el_type,
                "content": sub,
            })
            local_id += 1

    return chunks


def add_chunks(ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], documents: List[str]) -> None:
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )


def get_existing_ids() -> set:
    results = collection.get(include=[])
    return set(results["ids"]) if results and "ids" in results else set()


def normalize_chunk_id(cid: str) -> str:
    """
    Normalize chunk IDs to reduce duplicates caused by typos or odd chars.
    """
    cid = cid.strip().lower()
    cid = re.sub(r'\.+pdf', '.pdf', cid)          
    cid = re.sub(r'[^a-z0-9_.-]', '_', cid)       
    return cid

    
def embed_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 16,
    model: str = AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    checkpoint_file: Optional[str] = None
) -> None:
    """
    Embed chunks into ChromaDB with Azure OpenAI embeddings, resume-safe via atomic checkpoints.
    """
    if checkpoint_file is None and chunks:
        source_pdf = chunks[0]["source_pdf"]
        checkpoint_file = get_checkpoint_file(source_pdf, prefix="embed")  

    checkpoint = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = set(json.load(f))
        print(f"🔄 Resuming embeddings from checkpoint with {len(checkpoint)} IDs")

    existing_ids = get_existing_ids()
    print(f"ℹ️ Skipping {len(existing_ids)} chunks already present in Chroma")

    # Normalize and deduplicate
    seen = set()
    unique_chunks: List[Dict[str, Any]] = []
    for c in chunks:
        cid = normalize_chunk_id(c["chunk_id"])
        c["chunk_id"] = cid
        if cid not in seen:
            seen.add(cid)
            unique_chunks.append(c)
        else:
            print(f"⚠️ Duplicate chunk_id skipped before embedding: {cid}")

    # Filter out already embedded
    new_chunks = [c for c in unique_chunks if c["chunk_id"] not in existing_ids and c["chunk_id"] not in checkpoint]
    print(f"➡️ {len(new_chunks)} new chunks to embed")

    total_attempted = len(new_chunks)
    total_embedded = 0
    total_failed = 0

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i+batch_size]
        if not batch:
            continue

        texts = [c["content"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "source_pdf": c["source_pdf"],
            "page_number": c["page_number"],
            "type": c["type"],
            "section_heading": c.get("section_heading"),
        } for c in batch]

        # Retry with exponential backoff
        for attempt in range(3):
            try:
                resp = openai_client.embeddings.create(
                    model=model,
                    input=texts
                )
                vectors = [item.embedding for item in resp.data]

                add_chunks(ids, vectors, metadatas, texts)

                # Atomic checkpoint update
                checkpoint.update(ids)
                tmp_file = checkpoint_file + ".tmp"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(list(checkpoint), f, indent=2, ensure_ascii=False)
                os.replace(tmp_file, checkpoint_file)

                total_embedded += len(ids)
                print(f"✅ Embed batch {i//batch_size+1} complete; total checkpointed {len(checkpoint)}")
                break
            except Exception as e:
                print(f"⚠️ Embed batch {i//batch_size+1} failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"❌ Skipping embed batch {i//batch_size+1} after 3 failed attempts")
            total_failed += len(ids)

    print(f"\n🎉 Embedding summary: attempted {total_attempted}, embedded {total_embedded}, failed {total_failed}, skipped {len(existing_ids)} existing.")


def process_pdf(blob_name: str, loader: BlobLoader) -> None:
    print(f"\n▶️ Processing PDF: {blob_name}")
    blob_bytes = loader.download_blob_as_bytes(blob_name)

    # Text + tables
    elements = extract_text_and_tables_from_blob(blob_bytes)
    print(f"📝 Extracted {len(elements)} text/table elements")
    text_chunks = chunk_elements(elements, source_pdf=blob_name)
    print(f"🔪 Chunked into {len(text_chunks)} text/table chunks")

    # Images + captions
    images = extract_images_from_pdf(blob_bytes, blob_name)
    print(f"🖼️ Extracted {len(images)} images")
    caption_checkpoint = get_checkpoint_file(blob_name, prefix="captions")
    captions = batch_caption_images(images, batch_size=10, checkpoint_file=caption_checkpoint)
    print(f"💬 Generated {len(captions)} image captions")

    caption_chunks = [
        {
            "chunk_id": c["chunk_id"],
            "content": c["caption"],
            "source_pdf": c["source_pdf"],
            "page_number": c["page_number"],
            "type": "image_caption",
        }
        for c in captions
    ]

    # Combine
    all_chunks = text_chunks + caption_chunks
    print(f"📦 Total chunks for {blob_name}: {len(all_chunks)}")

    # Embeddings
    embed_checkpoint = get_checkpoint_file(blob_name, prefix="embed")
    embed_chunks(all_chunks, batch_size=16, checkpoint_file=embed_checkpoint)

    print(f"🎉 Ingestion complete for {blob_name}")

def ingest_all_pdfs() -> None:
    """Process all PDFs in the blob container automatically (deployment-ready)."""
    loader = BlobLoader()
    blobs = loader.list_blobs()
    if not blobs:
        print("❌ No PDFs found in container")
        return

    for blob_name in blobs:
        if blob_name.lower().endswith(".pdf"):
            process_pdf(blob_name, loader)
        else:
            print(f"⏭️ Skipping non-PDF blob: {blob_name}")

    print("✅ Ingestion complete for all PDFs")


if __name__ == "__main__":
    ingest_all_pdfs()

