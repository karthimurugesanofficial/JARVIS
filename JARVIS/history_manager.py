import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

HISTORY_FILE = "/Users/karthimurugesan/Desktop/JARVIS/conversation_history.json"

# Load semantic model once
model = SentenceTransformer("all-MiniLM-L6-v2")


# --- 🧠 Semantic Recall: Top-K relevant user queries ---
def get_relevant_past_messages(query, history, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Only consider past user messages with timestamps
    user_messages = [
        {"content": m["content"], "timestamp": m.get("timestamp", "N/A")}
        for m in history if m.get("role") == "user"
    ]

    if not user_messages:
        return []

    corpus = [m["content"] for m in user_messages]
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    top_indices = scores.argsort(descending=True)[:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "content": user_messages[idx]["content"],
            "timestamp": user_messages[idx]["timestamp"],
            "score": scores[idx].item()
        })
    
    return results


# --- 💾 Load full conversation history ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return []  # handle empty file
                return json.loads(content)
        except Exception as e:
            print("⚠️ Failed to load history:", e)
    return []



# --- 💾 Save updated conversation history ---
def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print("⚠️ Failed to save history:", e)
