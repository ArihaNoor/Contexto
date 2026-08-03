"""Short-term multi-turn chat memory, isolated per session."""

from langchain_core.chat_history import InMemoryChatMessageHistory

_histories: dict[str, InMemoryChatMessageHistory] = {}

MAX_TURNS = 10  # keep the last N question/answer pairs


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()
    return _histories[session_id]


def append_turn(session_id: str, question: str, answer: str) -> None:
    history = get_history(session_id)
    history.add_user_message(question)
    history.add_ai_message(answer)
    if len(history.messages) > MAX_TURNS * 2:
        history.messages = history.messages[-MAX_TURNS * 2 :]


def clear_history(session_id: str) -> None:
    _histories.pop(session_id, None)
