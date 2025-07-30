# query_preprocessor.py

from utils import get_current_location
from history_manager import get_relevant_past_messages  # Existing
from togenther_api import query_deepseek  # For LLM rephrasing

def preprocess_query(query: str, history: list = None) -> str:
    # Optional: Get location once (we'll let LLM decide if/how to use it)
    location = get_current_location()
    location_text = f"User's current location: {location['city']}, {location['region']}, {location['country']}" if location else ""

    # 👈 Fully Dynamic Enhancement: Use LLM to rephrase/ enhance for ALL queries if history exists
    if history:
        # Pull top 3 semantically relevant past messages
        relevant_msgs = get_relevant_past_messages(query, history, top_k=3)
        recent_context = "\n".join([f"{msg['timestamp']}: {msg['content']} (relevance: {msg['score']:.2f})" for msg in relevant_msgs]) if relevant_msgs else ""

        # LLM prompt for dynamic rephrasing (handles ambiguity, location, everything adaptively)
        rephrase_prompt = f"""
You are a query enhancer for an AI assistant. Analyze the recent conversation history and the current user query. If the query is ambiguous, a follow-up, or could be improved with context (e.g., adding location or inferring from previous topics), rewrite it as a clear, standalone question. Incorporate relevant details from history or location if it makes sense. If the query is already clear and complete, return it unchanged. Keep it natural and concise.

User location info (use only if relevant): {location_text}

Recent history (most relevant snippets):
{recent_context}

Current query: {query}

Enhanced query:
""".strip()

        try:
            enhanced_query = query_deepseek(rephrase_prompt).strip()
            if enhanced_query and enhanced_query.lower() != query.lower():
                query = enhanced_query
                print(f"🔄 Dynamically enhanced to: {query}")  # Debug log
        except Exception as e:
            print(f"⚠️ Enhancement failed: {e} — using original query.")

    return query
