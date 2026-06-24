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

# VectorAI collection names for each data source
FAQ_COLLECTION = "product_faq"
DOCS_COLLECTION = "product_docs"
TICKETS_COLLECTION = "resolved_tickets"
MEMORY_COLLECTION = "resolved_queries"
