"""Generation: build the RAG prompt and produce the answer."""
from tenacity import retry

from app.rag.llm import chat, retry_policy
from app.rag.retrieval import Result, fetch_context

SYSTEM_PROMPT = """
You are a knowledgeable, precise assistant for RISC-V technical specifications.
You are chatting with an engineer about RISC-V (ISA profiles, SBI, boot and runtime services, vector extension).
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so. Do not guess register names, encodings or requirements.
Cite the source document and page range for the facts you use, e.g. (rva23.pdf, pages 7-12).
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


def make_rag_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    context = "\n\n".join(
        f"Extract from {c.metadata.get('source', '?')} (pages {c.metadata.get('pages', '?')}):\n{c.page_content}"
        for c in chunks
    )
    return (
        [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(**retry_policy)
def _generate(messages: list[dict]) -> str:
    return chat(messages).choices[0].message.content


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Result]]:
    """RAG entry point used by the CLI now and by FastAPI in task 1.1.4."""
    history = history or []
    chunks = fetch_context(question, history)
    return _generate(make_rag_messages(question, history, chunks)), chunks
