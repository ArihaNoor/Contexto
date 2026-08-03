"""RAG pipeline: retrieve top-k chunks, inject context, generate a grounded answer."""

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.schemas import Source
from app.services import llm, memory
from app.services.vectorstore import get_vectorstore

SYSTEM_PROMPT = (
    "You are Contexto, an AI document assistant. Answer the user's question "
    "strictly using the provided context blocks. If the answer is not present "
    "in the context, explicitly state that the document does not contain "
    "enough information."
)

EXCERPT_LENGTH = 200


def answer_query(session_id: str, query: str) -> tuple[str, list[Source]]:
    retriever = get_vectorstore(session_id).as_retriever(
        search_kwargs={"k": settings.top_k}
    )
    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"[Context block {i} — page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in memory.get_history(session_id).messages:
        if isinstance(message, HumanMessage):
            messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            messages.append({"role": "assistant", "content": message.content})
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }
    )

    answer = llm.generate(messages)
    memory.append_turn(session_id, query, answer)

    sources = [
        Source(
            page=int(doc.metadata.get("page", 0)),
            excerpt=doc.page_content[:EXCERPT_LENGTH],
        )
        for doc in docs
    ]
    return answer, sources
