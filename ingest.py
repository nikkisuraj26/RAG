"""
ingest.py — Run this ONCE to chunk and embed your Gutenberg book into Chroma.
Usage: python ingest.py
"""

import os
import time
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-ingest")

BOOK_PATH  = os.getenv("BOOK_PATH", "data/The Psychopathology of Everyday Life.txt")
BOOK_TITLE = os.getenv("BOOK_TITLE", "The Psychopathology of Everyday Life.txt")
CHROMA_DIR = "./chroma_db"
COLLECTION = "gutenberg_rag"

# ── 1. LOAD ──────────────────────────────────────────────────
logger.info(f"Loading book: {BOOK_PATH}")
loader = TextLoader(BOOK_PATH, encoding="utf-8")
raw_docs = loader.load()
logger.info(f"Loaded {len(raw_docs)} document(s), total chars: {sum(len(d.page_content) for d in raw_docs)}")

# ── 2. CHUNK ─────────────────────────────────────────────────
logger.info("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(raw_docs)

# Enrich metadata on every chunk — this shows in Phoenix traces
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"]    = i
    chunk.metadata["book_title"]  = BOOK_TITLE
    chunk.metadata["source_file"] = BOOK_PATH
    chunk.metadata["char_count"]  = len(chunk.page_content)

logger.info(f"Total chunks created: {len(chunks)}")
logger.info(f"Avg chunk size: {sum(c.metadata['char_count'] for c in chunks) // len(chunks)} chars")
# ── 3. EMBED + STORE (RATE-LIMIT SAFE) ───────────────────────

logger.info("Embedding and storing in Chroma (rate-limit safe mode)...")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Initialize empty vector store
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# ✅ SAFE CONFIG (for S0 tier + ada-002)
BATCH_SIZE = 20       # ≤20 is safe
DELAY = 3             # seconds between batches
MAX_RETRIES = 5       # retry attempts


def add_with_retry(batch, batch_num):
    """Add documents with retry on rate limit"""
    for attempt in range(MAX_RETRIES):
        try:
            vector_store.add_documents(batch)
            logger.info(f"✅ Batch {batch_num} stored ({len(batch)} chunks)")
            return
        except Exception as e:
            logger.warning(f"⚠️ Batch {batch_num} failed (attempt {attempt+1}): {e}")

            if "429" in str(e) or "RateLimit" in str(e):
                logger.warning("⏳ Rate limit hit. Sleeping 60 seconds...")
                time.sleep(60)
            else:
                time.sleep(10)

    raise Exception(f"❌ Batch {batch_num} failed after {MAX_RETRIES} retries")


# ✅ BATCH PROCESSING
start = time.time()

total_batches = (len(chunks) // BATCH_SIZE) + 1
logger.info(f"Processing {len(chunks)} chunks in {total_batches} batches...")

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    batch_num = (i // BATCH_SIZE) + 1

    add_with_retry(batch, batch_num)

    # small delay to avoid burst traffic
    time.sleep(DELAY)

elapsed = round(time.time() - start, 2)

logger.info(f"✅ Ingestion complete! {len(chunks)} chunks stored in {CHROMA_DIR} ({elapsed}s)")
