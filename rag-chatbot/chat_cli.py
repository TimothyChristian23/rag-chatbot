"""
Simple CLI to test your RAG pipeline without starting the API.

Usage:
    python chat_cli.py
"""
from src.generation.chain import LEGAL_DISCLAIMER, build_rag_chain
from src.retrieval.vectorstore import load_vectorstore


def main():
    print("=== International Student OPT Assistant CLI ===")
    print(LEGAL_DISCLAIMER)
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
