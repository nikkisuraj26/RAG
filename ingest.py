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
    chunk_overlap=150,
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

# ── 3. EMBED + STORE ─────────────────────────────────────────
logger.info("Embedding and storing in Chroma (this may take a minute)...")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("OPENAI_DEPLOYMENT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

start = time.time()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION,
    persist_directory=CHROMA_DIR,
)
elapsed = round(time.time() - start, 2)

logger.info(f"✅ Ingestion complete! {len(chunks)} chunks stored in {CHROMA_DIR} ({elapsed}s)")
logger.info("Now run: python rag_agent_poc.py")