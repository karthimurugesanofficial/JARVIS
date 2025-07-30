# vector_memory.py

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from memory_manager import get_memory
from history_manager import get_relevant_past_messages, load_history

class VectorMemory:
    def __init__(self, persist_dir="/Users/karthimurugesan/Desktop/JARVIS/jarvis_memory"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="jarvis_memory")

    def add(self, text, role, timestamp):
        self.prune_irrelevant_entries()  # Trigger pruning before adding new entry
        embedding = self.model.encode([text])[0].tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"role": role, "timestamp": timestamp}],
            ids=[str(hash(text + timestamp))]
        )

    def prune_irrelevant_entries(self, max_size=500, relevance_threshold=0.3, history_relevance_threshold=0.4, protect_recent_days=7):
        """Prune irrelevant entries based on similarity to user's core profile and recent history."""
        # Check if pruning is needed
        if self.collection.count() <= max_size:
            print("🧹 Collection size is under limit—no pruning needed.")
            return
        
        current_time = datetime.now()
        protect_time = current_time - datetime.timedelta(days=protect_recent_days)
        
        # Get user's core profile for embedding
        profile = get_memory()
        profile_text = "\n".join([f"{k}: {v}" for k, v in profile.items()]) if profile else ""
        profile_embedding = self.model.encode([profile_text])[0] if profile_text else None
        
        # Load history and get a "history summary" embedding (e.g., top 3 relevant past messages concatenated)
        history = load_history()
        if history:
            # Use a dummy query like "summarize recent context" to get top relevant history
            recent_history = get_relevant_past_messages("summarize recent context", history, top_k=3)
            history_text = "\n".join([r["content"] for r in recent_history])
            history_embedding = self.model.encode([history_text])[0] if history_text else None
        else:
            history_embedding = None
        
        # If no profile or history, skip to avoid over-pruning
        if not profile_embedding and not history_embedding:
            print("🧹 No profile or history available—skipping prune.")
            return
        
        # Get all entries
        all_results = self.collection.get(include=['embeddings', 'metadatas', 'documents'])
        
        to_delete = []
        for i, (emb, meta, doc) in enumerate(zip(all_results['embeddings'], all_results['metadatas'], all_results['documents'])):
            if meta and 'timestamp' in meta:
                entry_time = datetime.fromisoformat(meta['timestamp'])
                is_recent = entry_time >= protect_time
                is_assistant = meta.get('role') == 'assistant'
                
                # Skip protected entries
                if is_recent or is_assistant:
                    continue
                
                entry_emb = np.array(emb).reshape(1, -1)
                
                # Calculate similarities
                profile_sim = cosine_similarity([profile_embedding], entry_emb)[0][0] if profile_embedding else 1.0  # Default high if no profile
                history_sim = cosine_similarity([history_embedding], entry_emb)[0][0] if history_embedding else 1.0  # Default high if no history
                
                # Prune if low on both (or adjust logic as needed)
                if profile_sim < relevance_threshold and history_sim < history_relevance_threshold:
                    to_delete.append(all_results['ids'][i])
                    print(f"🧹 Pruning irrelevant entry: '{doc[:50]}...' (profile sim: {profile_sim:.2f}, history sim: {history_sim:.2f})")
        
        if to_delete:
            self.collection.delete(ids=to_delete)
            print(f"🧹 Pruned {len(to_delete)} irrelevant entries.")
        else:
            print("🧹 No irrelevant entries found to prune.")

    def search(self, query, top_k=3):
        embedding = self.model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        return results["documents"][0] if results["documents"] else []
