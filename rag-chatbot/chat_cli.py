"""
Simple CLI to test your RAG pipeline without starting the API.

Usage:
    python chat_cli.py
"""
from src.retrieval.vectorstore import load_vectorstore
from src.generation.chain import build_rag_chain

def main():
    print("=== RAG Chatbot CLI ===")
    print("Loading vector store...")
    vs = load_vectorstore()
    chain = build_rag_chain(vs)
    print("Ready! Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        answer = chain.invoke(query)
        print(f"\nBot: {answer}\n")

if __name__ == "__main__":
    main()
