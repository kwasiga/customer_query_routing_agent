# app.py
#
# Minimal Streamlit UI for the customer query routing pipeline.
# Takes a customer query, runs it through retrieval -> triage -> resolution/escalation,
# and displays each stage's output. On a successful resolution, the result is written
# back into agent memory so future similar queries retrieve it as context.
#
# Run with: streamlit run app.py

from __future__ import annotations

import streamlit as st

from routing_agent.embedder import Embedder
from routing_agent.orchestrator import decide
from routing_agent.resolver import Resolver
from routing_agent.router import route
from routing_agent.vector_db import get_client, init_collections, seed_collections, write_memory

st.set_page_config(page_title="query router", page_icon="•", layout="centered")

st.markdown(
    """
    <style>
    * { font-family: "SF Mono", "Menlo", "Consolas", monospace !important; }
    [data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stButton button { border-radius: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="loading pipeline (embedder, vector db, llm)...")
def _load_pipeline():
    embedder = Embedder()
    client = get_client()
    init_collections(client)
    seed_collections(client, embedder)
    resolver = Resolver()
    return embedder, client, resolver


embedder, client, resolver = _load_pipeline()

st.title("customer query router")

query = st.text_area(
    "customer query",
    height=100,
    placeholder="e.g. My package says delivered but I never received it.",
)
submitted = st.button("route")

if submitted and query.strip():
    with st.spinner("retrieving context..."):
        routing = route(query, client, embedder)

    decision = decide(routing)

    st.divider()
    st.markdown(f"**department**  \n{routing.department}")

    if decision.should_resolve:
        st.markdown("**triage**  \n:green[RESOLVE]")
    else:
        st.markdown("**triage**  \n:red[ESCALATE]")
        for reason in decision.escalation_reasons:
            st.markdown(f"- {reason}")

    with st.expander(f"context used ({len(routing.context_docs)} docs)"):
        for doc in routing.context_docs:
            st.markdown(f"`{doc['score']:.2f}` **{doc['source_label']}** — {doc['department']}")
            st.text((doc["question"] or doc.get("title", "")) + "\n" + doc["answer"][:300])

    st.divider()

    with st.spinner("generating response..."):
        if decision.should_resolve:
            response = resolver.resolve(query, routing)
            write_memory(client, embedder.embed(query), routing.department, query, response)
        else:
            response = resolver.escalate(query, routing, decision)

    st.markdown("**response**")
    st.text(response)
elif submitted:
    st.warning("enter a query first")
