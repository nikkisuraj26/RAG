"""
rag_agent_poc.py — RAG with full Glass-Box Observability via Phoenix + OpenTelemetry
Same instrumentation style as agent_poc.py — MELT coverage for every RAG stage.

Run order:
  1. python ingest.py            (once)
  2. python -m phoenix.server.main serve
  3. python rag_agent_poc.py
  4. Open Phoenix: http://localhost:6006
"""

import os
import json
import time
import uuid
import logging
from typing import TypedDict, List, Optional

from fastapi import FastAPI, Header
import uvicorn
from dotenv import load_dotenv

# ── OpenTelemetry Core ────────────────────────────────────────────────────────
from opentelemetry import trace, metrics
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.logs import set_logger_provider
from opentelemetry.sdk.logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk.logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http.log_exporter import OTLPLogExporter

# ── OpenInference (AI-Aware Semantics) ────────────────────────────────────────
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# ── LangChain / LangGraph ─────────────────────────────────────────────────────
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

load_dotenv()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OTLP ENDPOINTS  (same as agent_poc.py)
# ═════════════════════════════════════════════════════════════════════════════
OTLP_ENDPOINT         = "http://127.0.0.1:6006/v1/traces"
OTLP_METRICS_ENDPOINT = "http://127.0.0.1:4318/v1/metrics"
OTLP_LOGS_ENDPOINT    = "http://127.0.0.1:4318/v1/logs"

BOOK_TITLE = os.getenv("BOOK_TITLE", "Gutenberg Book")
CHROMA_DIR = "./chroma_db"
COLLECTION = "gutenberg_rag"

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRACES (Phoenix)
# ═════════════════════════════════════════════════════════════════════════════
resource = Resource.create({"service.name": "rag-book-assistant"})

tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT))
)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("rag-book-assistant")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — METRICS
# ═════════════════════════════════════════════════════════════════════════════
metric_reader   = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=OTLP_METRICS_ENDPOINT))
meter_provider  = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter           = metrics.get_meter("rag-book-assistant")

# RAG-specific metrics
query_counter        = meter.create_counter("rag_queries_total",          description="Total RAG queries")
error_counter        = meter.create_counter("rag_errors_total",           description="Total RAG errors")
duration_histogram   = meter.create_histogram("rag_request_duration_seconds", description="End-to-end RAG latency")
retrieval_histogram  = meter.create_histogram("rag_retrieval_latency_ms",  description="Retrieval step latency")
chunks_histogram     = meter.create_histogram("rag_chunks_retrieved",      description="Number of chunks retrieved per query")
context_histogram    = meter.create_histogram("rag_context_chars",         description="Characters sent to LLM as context")
grounded_counter     = meter.create_counter("rag_grounded_answers_total",  description="Answers confirmed grounded in context")
hallucination_counter= meter.create_counter("rag_ungrounded_answers_total",description="Answers flagged as ungrounded / abstained")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOGS (correlated with traces)
# ═════════════════════════════════════════════════════════════════════════════
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter(endpoint=OTLP_LOGS_ENDPOINT))
)
set_logger_provider(logger_provider)

handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
logger  = logging.getLogger("rag-logger")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AUTO-INSTRUMENTORS (Glass-Box Nets)
# ═════════════════════════════════════════════════════════════════════════════
LangChainInstrumentor().instrument()       # Captures LLM calls, chains, embeddings
OpenAIInstrumentor().instrument()          # Captures Azure/OpenAI direct SDK calls
HTTPXClientInstrumentor().instrument()     # Captures async LLM HTTP packets

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LLM + EMBEDDINGS + VECTOR STORE
# ═════════════════════════════════════════════════════════════════════════════
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

logger.info(f"Initializing LLM with Azure Deployment: {os.getenv('OPENAI_DEPLOYMENT')}")
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("OPENAI_DEPLOYMENT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_key=api_key,
    temperature=0
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("OPENAI_DEPLOYMENT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_key=api_key,
)

logger.info(f"Loading Chroma vector store from: {CHROMA_DIR}")
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
logger.info("Vector store ready.")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LANGGRAPH STATE
# ═════════════════════════════════════════════════════════════════════════════
class RAGState(TypedDict):
    user_query:      str
    rewritten_query: str
    retrieved_chunks: List[dict]       # [{chunk_id, text, score_hint, source}]
    context:         str               # assembled context sent to LLM
    answer:          str
    grounded:        bool              # did LLM confirm answer is grounded?
    abstained:       bool              # did LLM say "not in document"?

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — LANGGRAPH NODES (each is a traced span)
# ═════════════════════════════════════════════════════════════════════════════

def query_rewrite_node(state: RAGState) -> dict:
    """
    GLASS BOX: Rewrites the raw user query into a cleaner retrieval query.
    Tracing this lets you see in Phoenix whether the rewrite improved retrieval.
    """
    with tracer.start_as_current_span("rag.query_rewrite") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute("rag.query.original", state["user_query"])
        span.set_attribute("input.value", state["user_query"])

        logger.info(f"Rewriting query: {state['user_query']}")

        prompt = (
            f"Rewrite the following question into a clear, standalone search query "
            f"optimized for semantic retrieval from a book. "
            f"Return ONLY the rewritten query, nothing else.\n\n"
            f"Question: {state['user_query']}"
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip()
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"Query rewrite failed, using original: {e}")
            rewritten = state["user_query"]   # fallback to original

        # GLASS BOX: See both versions in Phoenix
        span.set_attribute("rag.query.rewritten", rewritten)
        span.set_attribute("output.value", rewritten)
        logger.info(f"Rewritten query: {rewritten}")

        return {"rewritten_query": rewritten}


def retrieve_chunks_node(state: RAGState) -> dict:
    """
    GLASS BOX: Retrieves top-k chunks from Chroma.
    Phoenix shows chunk IDs, previews, sources, and retrieval latency.
    """
    with tracer.start_as_current_span("rag.retrieve_chunks") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
        span.set_attribute("rag.query.used_for_retrieval", state["rewritten_query"])
        span.set_attribute("input.value", state["rewritten_query"])

        logger.info(f"Retrieving chunks for: {state['rewritten_query']}")

        t_start = time.time()
        try:
            docs = retriever.invoke(state["rewritten_query"])
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            raise
        retrieval_ms = round((time.time() - t_start) * 1000, 2)

        retrieved = []
        for i, doc in enumerate(docs):
            chunk_data = {
                "chunk_id":    doc.metadata.get("chunk_id", i),
                "book_title":  doc.metadata.get("book_title", BOOK_TITLE),
                "source_file": doc.metadata.get("source_file", "unknown"),
                "char_count":  doc.metadata.get("char_count", len(doc.page_content)),
                "text":        doc.page_content,
                "preview":     doc.page_content[:200].replace("\n", " "),
            }
            retrieved.append(chunk_data)

            # GLASS BOX: Each chunk becomes a named event in the span's Events tab
            span.add_event(
                f"Chunk {i+1} Retrieved",
                attributes={
                    "chunk.id":      str(chunk_data["chunk_id"]),
                    "chunk.preview": chunk_data["preview"],
                    "chunk.chars":   chunk_data["char_count"],
                    "chunk.source":  chunk_data["source_file"],
                }
            )

        # GLASS BOX: Summary attributes visible in span detail
        chunk_ids = [str(c["chunk_id"]) for c in retrieved]
        span.set_attribute("rag.retrieval.k_requested",   5)
        span.set_attribute("rag.retrieval.k_returned",    len(retrieved))
        span.set_attribute("rag.retrieval.chunk_ids",     json.dumps(chunk_ids))
        span.set_attribute("rag.retrieval.latency_ms",    retrieval_ms)
        span.set_attribute("output.value",                json.dumps([c["preview"] for c in retrieved]))

        # Emit metrics
        retrieval_histogram.record(retrieval_ms)
        chunks_histogram.record(len(retrieved))

        logger.info(f"Retrieved {len(retrieved)} chunks in {retrieval_ms}ms | IDs: {chunk_ids}")
        return {"retrieved_chunks": retrieved}


def build_context_node(state: RAGState) -> dict:
    """
    GLASS BOX: Assembles retrieved chunks into the final context string for the LLM.
    Phoenix shows exactly what text was sent to the model.
    """
    with tracer.start_as_current_span("rag.build_context") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute("rag.context.chunks_used", len(state["retrieved_chunks"]))

        context_parts = []
        for i, chunk in enumerate(state["retrieved_chunks"], start=1):
            context_parts.append(
                f"[Source: {chunk['book_title']} | Chunk #{chunk['chunk_id']}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # GLASS BOX: Full context visible in Phoenix span
        span.set_attribute("rag.context.full_text",   context[:5000])   # Phoenix preview limit
        span.set_attribute("rag.context.char_count",  len(context))
        span.set_attribute("rag.context.chunk_ids",   json.dumps([str(c["chunk_id"]) for c in state["retrieved_chunks"]]))
        span.set_attribute("output.value",            context[:2000])

        # Metric: how large was the context window we sent?
        context_histogram.record(len(context))

        logger.info(f"Context assembled: {len(context)} chars from {len(state['retrieved_chunks'])} chunks")
        return {"context": context}


def generate_answer_node(state: RAGState) -> dict:
    """
    GLASS BOX: Calls the LLM with retrieved context and generates a grounded answer.
    Forces the LLM to cite chunk numbers and admit when context is insufficient.
    Phoenix shows the full prompt, raw response, and grounding verdict.
    """
    with tracer.start_as_current_span("rag.generate_answer") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        span.set_attribute("input.value",           state["user_query"])
        span.set_attribute("rag.query.original",    state["user_query"])
        span.set_attribute("rag.query.rewritten",   state["rewritten_query"])
        span.set_attribute("rag.llm.model",         os.getenv("OPENAI_DEPLOYMENT"))

        system_prompt = (
            "You are a helpful assistant that answers questions strictly based on "
            "the provided book excerpts. Rules:\n"
            "1. Answer ONLY using information from the provided context.\n"
            "2. Cite the chunk numbers you used (e.g. 'According to Chunk #42...').\n"
            "3. If the answer is NOT in the context, respond with exactly: "
            "'NOT_IN_BOOK: I could not find this in the provided excerpts.'\n"
            "4. Be concise but complete.\n"
        )

        user_prompt = (
            f"Context from '{BOOK_TITLE}':\n\n"
            f"{state['context']}\n\n"
            f"---\n"
            f"Question: {state['user_query']}\n\n"
            f"Answer:"
        )

        logger.info(f"Calling LLM for answer generation...")

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            raw_answer = response.content.strip()
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            raise

        # GLASS BOX: Detect abstention vs grounded answer
        abstained = raw_answer.startswith("NOT_IN_BOOK")
        grounded  = not abstained

        span.set_attribute("output.value",             raw_answer)
        span.set_attribute("rag.answer.raw",           raw_answer)
        span.set_attribute("rag.answer.grounded",      grounded)
        span.set_attribute("rag.answer.abstained",     abstained)
        span.set_attribute("rag.answer.char_count",    len(raw_answer))

        # GLASS BOX: Events tab shows the full prompt payload
        span.add_event("LLM Prompt Sent", attributes={
            "prompt.system": system_prompt,
            "prompt.user_truncated": user_prompt[:1000],
        })
        span.add_event("LLM Response Received", attributes={
            "response.text":      raw_answer[:500],
            "response.grounded":  str(grounded),
            "response.abstained": str(abstained),
        })

        # Emit grounding metrics
        if grounded:
            grounded_counter.add(1)
            logger.info("Answer is GROUNDED in retrieved context.")
        else:
            hallucination_counter.add(1)
            logger.warning("Answer ABSTAINED — query not answerable from context.")

        return {
            "answer":    raw_answer,
            "grounded":  grounded,
            "abstained": abstained,
        }


def evaluate_node(state: RAGState) -> dict:
    """
    GLASS BOX: Lightweight self-evaluation — asks LLM to verify its own answer
    is supported by the chunks. Shows faithfulness score in Phoenix.
    """
    with tracer.start_as_current_span("rag.evaluate_faithfulness") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EVALUATOR.value)
        span.set_attribute("rag.eval.input_answer",   state["answer"][:500])
        span.set_attribute("rag.eval.abstained",      state["abstained"])

        if state["abstained"]:
            span.set_attribute("rag.eval.faithfulness", "N/A — abstained")
            span.set_attribute("rag.eval.verdict",      "ABSTAINED")
            logger.info("Eval skipped — LLM abstained (no answer to evaluate).")
            return {}

        eval_prompt = (
            f"Given this context:\n{state['context'][:3000]}\n\n"
            f"And this answer:\n{state['answer']}\n\n"
            f"Is the answer fully supported by the context above? "
            f"Reply with ONLY one word: FAITHFUL or UNFAITHFUL."
        )

        try:
            verdict_response = llm.invoke([HumanMessage(content=eval_prompt)])
            verdict = verdict_response.content.strip().upper()
        except Exception as e:
            span.record_exception(e)
            verdict = "UNKNOWN"

        faithful = "FAITHFUL" in verdict

        # GLASS BOX: Faithfulness verdict visible as attribute in Phoenix
        span.set_attribute("rag.eval.faithfulness_verdict", verdict)
        span.set_attribute("rag.eval.faithful",             faithful)
        span.set_attribute("output.value",                  verdict)

        span.add_event("Faithfulness Verdict", attributes={
            "verdict":  verdict,
            "faithful": str(faithful),
        })

        if not faithful:
            span.set_status(StatusCode.ERROR, "Answer flagged as UNFAITHFUL by evaluator")
            logger.warning(f"FAITHFULNESS EVAL: {verdict} — potential hallucination!")
        else:
            logger.info(f"FAITHFULNESS EVAL: {verdict} ✅")

        return {}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ASSEMBLE LANGGRAPH
# ═════════════════════════════════════════════════════════════════════════════
builder = StateGraph(RAGState)
builder.add_node("query_rewrite",  query_rewrite_node)
builder.add_node("retrieve",       retrieve_chunks_node)
builder.add_node("build_context",  build_context_node)
builder.add_node("generate",       generate_answer_node)
builder.add_node("evaluate",       evaluate_node)

builder.set_entry_point("query_rewrite")
builder.add_edge("query_rewrite", "retrieve")
builder.add_edge("retrieve",      "build_context")
builder.add_edge("build_context", "generate")
builder.add_edge("generate",      "evaluate")
builder.add_edge("evaluate",      END)

rag_agent = builder.compile()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FASTAPI ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="RAG Book Assistant", description="Glass-box RAG with Phoenix Observability")

@app.get("/ask")
async def ask(
    query: str,
    user_id:    str = Header(default="anonymous-user"),
    session_id: str = Header(default_factory=lambda: str(uuid.uuid4())),
):
    start_time = time.time()
    query_counter.add(1, {"book": BOOK_TITLE})

    # ROOT SPAN — the top-level trace seen in Phoenix
    with tracer.start_as_current_span("rag.pipeline.root") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
        span.set_attribute("user.id",       user_id)
        span.set_attribute("session.id",    session_id)
        span.set_attribute("input.value",   query)
        span.set_attribute("rag.book",      BOOK_TITLE)
        span.set_attribute("rag.chroma_dir", CHROMA_DIR)

        logger.info(f"[{session_id}] Query from {user_id}: {query}")

        try:
            result = rag_agent.invoke({
                "user_query":       query,
                "rewritten_query":  "",
                "retrieved_chunks": [],
                "context":          "",
                "answer":           "",
                "grounded":         False,
                "abstained":        False,
            })

            # GLASS BOX: Final state bound to root span
            span.set_attribute("output.value",          result["answer"])
            span.set_attribute("rag.final.grounded",    result["grounded"])
            span.set_attribute("rag.final.abstained",   result["abstained"])
            span.set_attribute("rag.final.chunks_used", len(result["retrieved_chunks"]))
            span.set_attribute("rag.final.context_chars", len(result["context"]))

            logger.info(f"[{session_id}] Answer ready. Grounded={result['grounded']}")

            return {
                "query":         query,
                "rewritten":     result["rewritten_query"],
                "answer":        result["answer"],
                "grounded":      result["grounded"],
                "abstained":     result["abstained"],
                "chunks_used":   len(result["retrieved_chunks"]),
                "chunk_ids":     [c["chunk_id"] for c in result["retrieved_chunks"]],
                "book":          BOOK_TITLE,
            }

        except Exception as e:
            error_counter.add(1, {"book": BOOK_TITLE, "error_type": type(e).__name__})
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"[{session_id}] RAG pipeline error: {e}", exc_info=True)
            raise

        finally:
            duration = time.time() - start_time
            duration_histogram.record(duration)
            logger.info(f"[{session_id}] Request completed in {round(duration, 2)}s")


@app.get("/health")
async def health():
    return {"status": "ok", "book": BOOK_TITLE, "chroma_dir": CHROMA_DIR}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

# ──────────────────────────────────────────────────────────────────────────────
# DEMO CURL COMMANDS (paste in a new terminal after all 3 services are running)
# ──────────────────────────────────────────────────────────────────────────────
# Good retrieval:
#   curl "http://localhost:8001/ask?query=How does Freud connect everyday mistakes to the unconscious mind?"
#
# Multi-chunk:
#   curl "http://localhost:8001/ask?query=What does Freud say about forgetting proper names?"
#
# Abstention (out-of-scope):
#   curl "http://localhost:8001/ask?query=How does Freud connect everyday mistakes to the unconscious mind?"
#
# Character detail:
#   curl "http://localhost:8001/ask?query=What is Freud’s opinion on modern smartphone addiction?"