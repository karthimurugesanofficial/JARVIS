from memory_updater import extract_and_update
from query_router import route_query
from query_preprocessor import preprocess_query
from history_manager import load_history, save_history
from vector_memory import VectorMemory
from datetime import datetime
from prompt_builder import build_prompt
from togenther_api import query_deepseek
from memory_manager import get_memory

# 🎯 Initialize
memory = VectorMemory()
conversation_history = load_history()

print("Hello! I’m Jarvis. Ask me anything.")

# 🔁 Main loop
while True:
    query = input("You: ").strip()
    if query.lower() in ['exit', 'quit']:
        save_history(conversation_history)
        print("💾 Conversation saved. Goodbye!")
        break

    # 🧼 Preprocess query (adds location context if needed)
    query = preprocess_query(query)
    timestamp = datetime.now().isoformat()

    # 🧾 Save user input to history and Chroma (this triggers forgetting via VectorMemory.add())
    conversation_history.append({"role": "user", "content": query, "timestamp": timestamp})
    memory.add(query, role="user", timestamp=timestamp)

    # 👈 New: Route the query through query_router for intent handling and proactive suggestions
    intent, response = route_query(query, conversation_history)

    # 👈 New: Fallback to your original prompt-building if it's a general/LLM intent
    if intent in ["general", "llm"]:  # Adjust based on your classify_query_llm labels
        # 🔁 🔍 Retrieve top 5 relevant past conversation chunks
        recalled_convo = memory.search(query, top_k=5)
        context_text = "\n".join(recalled_convo)

        # 🧠 Inject personal memory
        profile_memory = get_memory()
        memory_text = "\n".join([f"{k}: {v}" for k, v in profile_memory.items()])

        # 🧠 Build dynamic prompt
        prompt = build_prompt(context_text, memory_text, query)

        # 🤖 Query LLM
        response = query_deepseek(prompt)

    # ✅ Save assistant response to history and Chroma (triggers forgetting again if needed)
    conversation_history.append({"role": "assistant", "content": response, "timestamp": timestamp})
    memory.add(response, role="assistant", timestamp=timestamp)

    print("Jarvis (🧠):", response)

    # 💾 Save history
    save_history(conversation_history)

    # 🧠 Update memory if needed
    if any(word in query.lower() for word in ["change", "update", "set", "my name is", "my goal is"]):
        update_result = extract_and_update(query)
        if update_result:
            print("🧠", update_result)
