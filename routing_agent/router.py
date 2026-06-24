# router.py
#
# Core routing logic for the customer query agent.
# Takes a raw customer query, searches all 4 VectorAI collections for relevant context,
# and returns a unified list of ranked documents to pass to the LLM.
# Contains:
#   - SOURCE_LABELS       : human-readable labels for each data source
#   - _hits_to_docs()     : converts raw VectorAI search hits into a standard dict format
#   - route()             : embeds the query, searches all collections, returns context docs

from __future__ import annotations

from actian_vectorai import VectorAIClient

from routing_agent.embedder import Embedder
from routing_agent.vector_db import (
    search_docs,
    search_faqs,
    search_memory,
    search_tickets,
)

# Maps internal source keys to display labels shown in the UI or LLM prompt
SOURCE_LABELS = {
    "faq": "FAQ",
    "docs": "Policy Doc",
    "tickets": "Past Ticket",
    "memory": "Agent Memory",
}


def _hits_to_docs(hits: list, source_key: str) -> list[dict]:
    # Converts raw VectorAI ScoredPoint results into a standardised dict format.
    # Handles different payload field names across collections (e.g. "question" vs "summary").
    docs = []
    for hit in hits:
        p = hit.payload
        docs.append({
            "source": source_key,
            "source_label": SOURCE_LABELS[source_key],
            "department": p.get("department", ""),
            "question": p.get("question") or p.get("summary") or p.get("query") or p.get("title", ""),
            "answer": p.get("answer") or p.get("content") or p.get("thread") or p.get("resolution", ""),
            "score": round(float(hit.score), 3),
            "ticket_id": p.get("ticket_id", ""),
            "resolution_type": p.get("resolution_type", ""),
            "doc_type": p.get("doc_type", ""),
            "title": p.get("title", ""),
        })
    return docs


def route(query: str, client: VectorAIClient, embedder: Embedder) -> list[dict]:
    # Embeds the customer query and searches all 4 collections in parallel.
    # Returns a combined list of context documents sorted by relevance score.
    query_vector = embedder.embed(query)

    faq_hits = search_faqs(client, query_vector, top_k=3)
    doc_hits = search_docs(client, query_vector, top_k=2)
    ticket_hits = search_tickets(client, query_vector, top_k=2)
    memory_hits = search_memory(client, query_vector, top_k=2)

    all_docs = (
        _hits_to_docs(faq_hits, "faq")
        + _hits_to_docs(doc_hits, "docs")
        + _hits_to_docs(ticket_hits, "tickets")
        + _hits_to_docs(memory_hits, "memory")
    )

    return sorted(all_docs, key=lambda d: d["score"], reverse=True)
