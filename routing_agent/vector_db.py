# vector_db.py
#
# Handles all interaction with the Actian VectorAI database.
# Contains:
#   - get_client()        : factory that builds and returns a VectorAIClient
#   - init_collections()  : creates the 4 vector collections if they don't exist
#   - seed_collections()  : loads FAQ, docs, and ticket data into VectorAI on first run
#   - search_faqs()       : searches the FAQ collection by vector similarity
#   - search_docs()       : searches the policy docs collection by vector similarity
#   - search_tickets()    : searches resolved support tickets by vector similarity
#   - search_memory()     : searches the agent memory collection by vector similarity
#   - write_memory()      : saves a resolved query into the agent memory collection

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from routing_agent.embedder import Embedder

from actian_vectorai import VectorAIClient
from actian_vectorai.models import Distance, PointStruct, VectorParams

from routing_agent.config import (
    DOCS_COLLECTION,
    EMBEDDING_DIMENSION,
    FAQ_COLLECTION,
    MEMORY_COLLECTION,
    TICKETS_COLLECTION,
    VECTORAI_ACCESS_TOKEN,
    VECTORAI_URL,
)


def get_client() -> VectorAIClient:
    # Builds a VectorAIClient using the URL and optional access token from config.
    # Only passes the token if one is set — avoids sending a None value to the client.
    kwargs: dict = {"url": VECTORAI_URL}
    if VECTORAI_ACCESS_TOKEN:
        kwargs["access_token"] = VECTORAI_ACCESS_TOKEN
    return VectorAIClient(**kwargs)


def init_collections(client: VectorAIClient) -> None:
    # Creates all 4 collections in VectorAI if they don't already exist.
    # Uses cosine similarity with 384 dimensions to match the embedding model output.
    existing = set(client.collections.list())
    vector_params = VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.Cosine)

    for collection in [FAQ_COLLECTION, DOCS_COLLECTION, TICKETS_COLLECTION, MEMORY_COLLECTION]:
        if collection not in existing:
            client.collections.create(collection, vectors_config=vector_params)


def seed_collections(client: VectorAIClient, embedder: Embedder, data_dir: str = "data") -> None:
    # Loads FAQs, docs, and tickets from JSON files into VectorAI on first run.
    # Skips any collection that already has data so it's safe to call on every startup.
    data_path = Path(data_dir)

    sources = [
        (FAQ_COLLECTION, data_path / "faqs.json",    lambda r: r["question"]),
        (DOCS_COLLECTION, data_path / "docs.json",   lambda r: r["title"] + " " + r["content"]),
        (TICKETS_COLLECTION, data_path / "tickets.json", lambda r: r["summary"]),
    ]

    for collection, file_path, text_fn in sources:
        if client.points.count(collection) > 0:
            continue

        records = json.loads(file_path.read_text())
        texts = [text_fn(r) for r in records]
        vectors = embedder._embed_batch(texts)

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=record)
            for record, vector in zip(records, vectors)
        ]
        client.points.upsert(collection, points=points)


def search_faqs(client: VectorAIClient, vector: list[float], top_k: int = 3) -> list:
    # Returns the top_k most similar FAQ entries to the given query vector.
    return client.points.search(FAQ_COLLECTION, vector=vector, limit=top_k)


def search_docs(client: VectorAIClient, vector: list[float], top_k: int = 2) -> list:
    # Returns the top_k most similar policy documents to the given query vector.
    return client.points.search(DOCS_COLLECTION, vector=vector, limit=top_k)


def search_tickets(client: VectorAIClient, vector: list[float], top_k: int = 2) -> list:
    # Returns the top_k most similar resolved support tickets to the given query vector.
    return client.points.search(TICKETS_COLLECTION, vector=vector, limit=top_k)


def search_memory(client: VectorAIClient, vector: list[float], top_k: int = 2) -> list:
    # Returns the top_k most similar past agent resolutions to the given query vector.
    return client.points.search(MEMORY_COLLECTION, vector=vector, limit=top_k)


def write_memory(
    client: VectorAIClient,
    vector: list[float],
    department: str,
    query: str,
    resolution: str,
) -> None:
    # Saves a resolved customer query into the agent memory collection.
    # Assigns a UUID so each memory entry is uniquely identifiable.
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "department": department,
            "query": query,
            "resolution": resolution,
        },
    )
    client.points.upsert(MEMORY_COLLECTION, points=[point])
