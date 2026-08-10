"""
RAG chain for an OPT-focused international student assistant.
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

LEGAL_DISCLAIMER = (
    "This information is for general education only and is not legal advice. "
    "Students should confirm their situation with their DSO or a qualified "
    "immigration attorney."
)

SYSTEM_PROMPT = """You are an OPT-focused assistant for international students in the United States.
You help students understand OPT, STEM OPT, CPT, work authorization, school reporting,
and common F-1 practical training questions.

Rules:
- Use ONLY the context provided below.
- If the answer is not supported by the context, say you do not have enough information.
- Do not invent deadlines, eligibility rules, forms, fees, government policies, or legal conclusions.
- Always include source citations from the retrieved context.
- When a question depends on a student's exact immigration record, school policy, or legal strategy,
  tell the student to contact their DSO or a qualified immigration attorney.
- Keep the tone clear, practical, and student-friendly.
- Include this disclaimer in every answer: {legal_disclaimer}

Context:
{context}
"""


def format_context(docs: list) -> str:
    """Merge retrieved chunks into a single context string with source labels."""
    if not docs:
        return "No retrieved context."

    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def collect_sources(docs: list) -> list[str]:
    """Return sorted unique source names from retrieved documents."""
    return sorted({doc.metadata.get("source", "unknown") for doc in docs})


def build_answer_chain():
    """Return a runnable chain that answers from an already-formatted context."""
    llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()


def generate_answer(question: str, docs: list, answer_chain=None) -> str:
    """Generate an answer from the exact documents that will be cited."""
    chain = answer_chain or build_answer_chain()
    return chain.invoke({
        "context": format_context(docs),
        "question": question,
        "legal_disclaimer": LEGAL_DISCLAIMER,
    })


def build_rag_chain(vectorstore):
    """Return a runnable RAG chain for CLI usage."""
    top_k = int(os.getenv("TOP_K_RESULTS", 5))
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    answer_chain = build_answer_chain()

    chain = (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough(),
            "legal_disclaimer": lambda _: LEGAL_DISCLAIMER,
        }
        | answer_chain
    )
    return chain
