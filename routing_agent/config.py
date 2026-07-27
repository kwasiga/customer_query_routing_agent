# config.py
#
# Central configuration for the routing agent.
# Loads all settings from environment variables (.env file).
# Contains:
#   - Embedding model name and vector dimension
#   - VectorAI connection settings (URL and optional access token)
#   - List of supported departments
#   - Number of context documents to retrieve (TOP_K)
#   - Collection names for each data source in VectorAI
#   - Local LLM settings (GGUF model source and inference parameters)

import os
from dotenv import load_dotenv

load_dotenv()

# Embedding model used to convert text into vectors
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Must match the output size of the embedding model above
EMBEDDING_DIMENSION = 384

# VectorAI connection — defaults work when running locally via Docker
VECTORAI_URL = os.getenv("ACTIAN_VECTORAI_URL")
# Only required if VectorAI auth is enabled; leave unset for local dev
VECTORAI_ACCESS_TOKEN = os.getenv("ACTIAN_VECTORAI_ACCESS_TOKEN")

# Supported departments queries can be routed to
DEPARTMENTS = [
    "Returns & Refunds",
    "Billing & Payments",
    "Technical Support",
    "Order Tracking",
    "General Inquiry",
]

# Number of context documents to retrieve per collection and pass to the LLM
TOP_K = 4

# Minimum top context-doc similarity score for the orchestrator to auto-resolve.
# Below this, the query is escalated to a human regardless of how the LLM would respond.
RESOLUTION_CONFIDENCE_THRESHOLD = float(os.getenv("RESOLUTION_CONFIDENCE_THRESHOLD", "0.55"))

# VectorAI collection names for each data source
FAQ_COLLECTION = "product_faq"
DOCS_COLLECTION = "product_docs"
TICKETS_COLLECTION = "resolved_tickets"
MEMORY_COLLECTION = "resolved_queries"

# Local LLM — GGUF-quantized Mistral model, run via llama-cpp-python
# Downloaded from the Hugging Face Hub on first use and cached under LLM_MODEL_DIR
LLM_REPO_ID = os.getenv("LLM_REPO_ID", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
LLM_FILENAME = os.getenv("LLM_FILENAME", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
LLM_MODEL_DIR = os.getenv("LLM_MODEL_DIR", "models")

LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# -1 = offload all layers to GPU (Metal on macOS); set to 0 to force CPU-only
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))
