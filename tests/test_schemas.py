import pytest
from pydantic import ValidationError

from app.api.schemas import ChatRequest, RetrieveRequest, Source, MAX_QUESTION_CHARS
from app.rag.retrieval import Result


def test_valid_chat_request_strips_whitespace_and_defaults_history():
    req = ChatRequest(question="  Is V mandatory in RVA23?  ")
    assert req.question == "Is V mandatory in RVA23?"
    assert req.history == []


@pytest.mark.parametrize("payload", [
    {"question": ""},                                                  # empty
    {"question": "   "},                                               # only spaces
    {"question": "x" * (MAX_QUESTION_CHARS + 1)},                      # too long
    {"question": "hi", "history": [{"role": "system", "content": "x"}]},  # role not allowed
    {"question": "hi", "temperature": 2},                              # unknown field
    {"question": "hi", "history": [{"role": "user", "content": "x"}] * 21},  # history too long
])
def test_invalid_chat_requests_are_rejected(payload):
    with pytest.raises(ValidationError):
        ChatRequest(**payload)


def test_retrieve_request_top_k_bounds():
    assert RetrieveRequest(query="sbi").top_k == 5
    with pytest.raises(ValidationError):
        RetrieveRequest(query="sbi", top_k=100)


def test_source_from_chunk_truncates_and_keeps_metadata():
    chunk = Result(page_content="a" * 500, metadata={"source": "rva23.pdf", "pages": "1-6"})
    s = Source.from_chunk(chunk)
    assert (s.source, s.pages, len(s.snippet)) == ("rva23.pdf", "1-6", 300)
