from app.rag.retrieval import Result, apply_order, merge_chunks


def test_apply_order_drops_invalid_and_duplicates_and_appends_skipped():
    assert apply_order([3, 3, 99, 1], ["a", "b", "c", "d"]) == ["c", "a", "b", "d"]


def test_merge_chunks_deduplicates_by_text():
    a = Result(page_content="x", metadata={})
    b = Result(page_content="y", metadata={})
    assert [c.page_content for c in merge_chunks([a], [a, b, b])] == ["x", "y"]
