from memory_manager import get_memory
from togenther_api import query_deepseek
from web_search import unified_search
from summarizer import summarize_web_result, summarize_today_conversation
from datetime import datetime
from utils import get_current_location
from history_manager import get_relevant_past_messages
from rag_engine import PersonalRAG
import random  # 👈 New import for random suggestion triggering

# --- 🔍 Dynamic LLM-Based Intent Classification ---
def classify_query_llm(user_input: str, history: list = None):
    recent_messages = history[-5:] if history else []
    recent_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

    prompt = f"""
You are a smart assistant. Based on the current message and recent conversation, classify the **user's intent** as one of the following:

- personal: related to user's identity, preferences, work, goals, etc.
- web_search: asking for external facts, people, places, current events.
- location: about user's current location or time.
- general: casual chat, feedback, or general-purpose input.

Recent chat history:
{recent_text}

Current user input:
User: {user_input}

Return just the label (e.g., personal, web_search, location, general).
""".strip()

    return query_deepseek(prompt)  # Or your LLM interface

# --- 🕰️ First User Query Retriever (optional) ---
def get_first_user_query(history):
    for msg in history:
        if msg.get("role") == "user":
            timestamp = msg.get("timestamp", "unknown time")
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime("%B %d, %Y at %I:%M %p")
            except:
                formatted_time = timestamp
            return msg["content"], formatted_time
    return None, None

# 👈 New: Function for generating proactive suggestions
def generate_proactive_suggestion(memory):
    """Generate a suggestion based on user's stored goals or preferences."""
    if not memory or 'goal' not in memory:
        return None
    
    goal = memory.get('goal', '').lower()
    if 'ai' in goal or 'expert' in goal:  # Example: Tailor to common goals
        suggestions = [
            "Based on your goal to become an AI expert, how about checking out the latest courses on Coursera?",
            "Remember your AI goal? I found a great article on recent AI advancements—want me to summarize it?"
        ]
        return random.choice(suggestions)
    # Add more conditions, e.g., for 'location': suggest local events
    return None

# --- 🧠 Route Query Based on Intent ---
def route_query(user_input, conversation_history):
    intent = classify_query_llm(user_input)
    print(f"📥 LLM classified as: {intent}")

    # --- 📌 Recall Previous Messages ---
    if intent == "recall":
        print("🧠 Detected recall query")

        # 🔁 Dynamically adjust top_k based on user query
        lowered = user_input.lower()
        if any(kw in lowered for kw in ["first question", "initial query", "start of chat"]):
            top_k = 1
        elif any(kw in lowered for kw in ["all", "entire", "everything", "full"]):
            top_k = 10
        elif any(kw in lowered for kw in ["summarize", "summary", "gist"]):
            top_k = 3
        else:
            top_k = 2  # default fallback

        results = get_relevant_past_messages(user_input, conversation_history, top_k=top_k)

        if not results:
            return "memory", "I couldn't find anything in our past chats related to that."

        formatted = "\n".join(
            [f'🕒 {r["timestamp"]} — "{r["content"]}" (score: {r["score"]:.2f})' for r in results]
        )
        return "memory", f"Here are your most relevant past questions (top {top_k}):\n\n{formatted}"

    elif intent == "self_reflect":
        print("🧠 Triggering self-analysis...")

        last_few = conversation_history[-6:]  # last 3 pairs (user + assistant)

        prompt = [
            {"role": "system", "content": "You are an AI assistant analyzing your recent performance."},
            {"role": "user", "content": (
                f"Here is the recent chat:\n\n{last_few}\n\n"
                "Analyze if you maintained continuity, responded relevantly, and used memory properly. "
                "What could be improved in terms of memory, coherence, or context awareness?"
            )}
        ]
        response = query_deepseek(prompt)
        return "llm", response

    # --- 📍 Location from IP ---
    elif intent == "location":
        print("📍 Detected location query — using IP-based location")
        location = get_current_location()
        if isinstance(location, dict):
            response = f"Based on your IP, you're currently in {location.get('city', 'Unknown City')}, {location.get('region', '')}, {location.get('country', '')}."
            return "memory", response
        else:
            print("🌐 IP lookup failed — using web fallback")
            web_snippets = unified_search(user_input)
            if not web_snippets:
                return "web", "Sorry, I couldn't detect your location."
            return "web", summarize_web_result(user_input, web_snippets)

    # --- 🧠 Personal Profile Answering ---
    elif intent == "personal":
        print("🧠 Detected personal query — using memory & LLM")

        # Try RAG-based memory retrieval first
        rag = PersonalRAG()
        rag_result = rag.query(user_input, top_k=3, threshold=0.7)

        # If RAG returns useful answer, use it
        if "couldn't find" not in rag_result:
            return "rag", rag_result

        # Else fallback to LLM with memory
        memory = get_memory()
        memory_text = "\n".join([f"{k}: {v}" for k, v in memory.items()])
        prompt = f"""You are Jarvis, the user's personal assistant.

User's memory:
{memory_text}

User asked: "{user_input}"

Reply helpfully using the above information if relevant."""
        response = query_deepseek([
            {"role": "system", "content": "You are Jarvis, the user's personal assistant."},
            {"role": "user", "content": prompt}
        ])
        return "llm", response

    # --- 🌐 Web Search for Live Info ---
    elif intent == "web_search":
        print("🌐 Detected web search query")
        web_snippets = unified_search(user_input)

        # Optional: fallback to RAG if web fails
        if not web_snippets or len(web_snippets.strip()) < 20:
            print("🌐 Web search failed — using fallback memory")
            rag = PersonalRAG()
            return "rag", rag.query(user_input)

        prompt = f"""User asked: {user_input}
Web search returned the following:
{web_snippets}

Answer clearly and briefly using this information."""
        response = query_deepseek([
            {"role": "system", "content": "You are Jarvis, a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        return "web", response

    # --- 🤖 Fallback: General Intent → Inject Context Summary ---
    else:
        print("🧠 Generic query — routing to conversation context with daily summary")
        today_summary = summarize_today_conversation(conversation_history)

        messages = [
            {"role": "system", "content": "You are Jarvis, the user's assistant."},
            {"role": "user", "content": f"Here is what we discussed today:\n\n{today_summary}\n\nNow reply to: {user_input}"}
        ]
        response = query_deepseek(messages)
        return "llm", response

    # 👈 New: Add proactive suggestion after generating response (for all intents)
    suggestion = generate_proactive_suggestion(get_memory())
    if suggestion and random.random() < 0.2:  # Suggest ~20% of the time to avoid overwhelming
        response += f"\n\n📍 Proactive tip: {suggestion}"
    return intent, response  # Updated to return intent (was intent_type in some spots)
