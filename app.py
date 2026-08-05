# app.py
#
# Streamlit UI for the customer query routing pipeline.
# Takes a customer query, runs it through retrieval -> triage -> resolution/escalation,
# and displays each stage's output. On a successful resolution, the result is written
# back into agent memory so future similar queries retrieve it as context.
#
# Run with: streamlit run app.py

from __future__ import annotations

import streamlit as st

from routing_agent.config import RESOLUTION_CONFIDENCE_THRESHOLD
from routing_agent.embedder import Embedder
from routing_agent.orchestrator import decide
from routing_agent.resolver import Resolver
from routing_agent.router import route
from routing_agent.vector_db import get_client, init_collections, seed_collections, write_memory

st.set_page_config(page_title="Custormer Support", page_icon="🎧", layout="wide")

SAMPLE_QUERIES = [
    "My Kubernetes cluster metrics are showing zero after upgrading to 1.28.",
    "I was charged twice for my annual renewal — can I get one of them refunded?",
    "My package says delivered but I never received it, and none of your docs explain what to do.",
]

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1100px; }
    [data-testid="stChatMessage"] { border-radius: 12px; }
    [data-testid="stSidebar"] button p { text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading pipeline (embedder, vector db, LLM)…")
def _load_pipeline():
    embedder = Embedder()
    client = get_client()
    init_collections(client)
    seed_collections(client, embedder)
    resolver = Resolver()
    return embedder, client, resolver


embedder, client, resolver = _load_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""


def run_query(query: str) -> None:
    with st.status("Routing query…", expanded=True) as status:
        st.write("🔎 Searching FAQs, policy docs, past tickets, and agent memory…")
        routing = route(query, client, embedder)
        st.write(f"Found **{len(routing.context_docs)}** relevant documents · routed to **{routing.department}**")

        status.update(label="Triaging…")
        decision = decide(routing)
        if decision.should_resolve:
            st.write("✅ Confidence high enough to auto-resolve")
        else:
            st.write("🚩 Escalating to a human agent")
            for reason in decision.escalation_reasons:
                st.write(f"- {reason}")

        status.update(label="Generating response…")
        if decision.should_resolve:
            response = resolver.resolve(query, routing)
            write_memory(client, embedder.embed(query), routing.department, query, response)
        else:
            response = resolver.escalate(query, routing, decision)

        status.update(label="Done", state="complete", expanded=False)

    st.session_state.history.insert(
        0, {"query": query, "routing": routing, "decision": decision, "response": response}
    )


with st.sidebar:
    st.markdown("### 🎧 How it works")
    st.markdown(
        "1. **Retrieve** — search FAQs, policy docs, past tickets & agent memory\n"
        "2. **Triage** — auto-resolve if retrieval confidence clears the bar, else escalate\n"
        "3. **Respond** — generate a grounded answer, or a warm handoff for a human"
    )
    st.caption(f"Resolution confidence threshold: {RESOLUTION_CONFIDENCE_THRESHOLD:.0%}")

    st.divider()
    st.markdown("### Try a sample query")
    for sample in SAMPLE_QUERIES:
        if st.button(sample, use_container_width=True, key=f"sample-{sample}"):
            st.session_state.query_input = sample
            st.session_state.autosubmit = True
            st.rerun()

    if st.session_state.history:
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

st.title("🎧 Customer Support")
st.caption(
    "An autonomous pipeline that retrieves context, triages confidence, "
    "and resolves or escalates customer queries."
)

query = st.text_area(
    "Customer query",
    key="query_input",
    height=90,
    placeholder="e.g. My package says delivered but I never received it.",
    label_visibility="collapsed",
)
submitted = st.button("Route query →", type="primary", use_container_width=True)

if st.session_state.pop("autosubmit", False):
    submitted = True

if submitted:
    if st.session_state.query_input.strip():
        run_query(st.session_state.query_input.strip())
    else:
        st.warning("Enter a query first.")

for entry in st.session_state.history:
    routing = entry["routing"]
    decision = entry["decision"]
    response = entry["response"]

    with st.chat_message("user"):
        st.write(entry["query"])

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.badge(routing.department, color="blue")
            if decision.should_resolve:
                st.badge("Resolved", icon="✅", color="green")
            else:
                st.badge("Escalated", icon="🚩", color="orange")
        with col2:
            top_score = routing.context_docs[0]["score"] if routing.context_docs else 0.0
            st.metric("Top match confidence", f"{top_score:.0%}")

        if not decision.should_resolve:
            for reason in decision.escalation_reasons:
                st.caption(f"⚠️ {reason}")

        if routing.context_docs:
            with st.expander(f"Context used ({len(routing.context_docs)} docs)"):
                for i, doc in enumerate(routing.context_docs):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(f"**{doc['source_label']}** · {doc['department']}")
                        st.caption(doc["question"] or doc.get("title", ""))
                    with c2:
                        st.progress(doc["score"], text=f"{doc['score']:.2f}")
                    st.text(doc["answer"][:300])
                    if i < len(routing.context_docs) - 1:
                        st.divider()

    avatar = "✅" if decision.should_resolve else "🚩"
    with st.chat_message("assistant", avatar=avatar):
        st.write(response)

    st.divider()
