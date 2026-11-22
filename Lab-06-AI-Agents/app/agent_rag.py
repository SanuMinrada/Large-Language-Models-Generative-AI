"""Lab 6 - Part C: RAG-powered AI Agent.

Three tools: Calculator, Memory Search, and RAG Search (reusing Lab 5's
vector store). Try: "What is NLP?" or "What is LoRA?" (same questions
tested in Lab 5) to confirm retrieval still works through the agent.
"""
from agent_core import chat_loop
from tools_basic import calculator
from tools_memory import search_student_memory, extract_and_save_memory
from tools_rag import search_university_knowledge

SYSTEM_PROMPT = (
    "You are a helpful university assistant with three tools: calculator, "
    "search_student_memory (recall things the user told you earlier), and "
    "search_university_knowledge (look up facts from university course material). "
    "Use search_university_knowledge for any factual/course-content question."
)

if __name__ == "__main__":
    chat_loop(
        SYSTEM_PROMPT,
        [calculator, search_student_memory, search_university_knowledge],
        "RAG-POWERED AGENT",
        on_user_message=extract_and_save_memory,
    )
