"""
RAG chain: takes a user query + retrieved context chunks,
builds a prompt, and returns the LLM's answer with source citations.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY
the context provided below. If the answer is not in the context, say
"I don't have enough information to answer that."

Always cite which document/source your answer comes from.

Context:
{context}
"""


def format_context(docs: list) -> str:
    """Merge retrieved chunks into a single context string with source labels."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(vectorstore):
    """Return a runnable RAG chain that retrieves context then generates an answer."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
