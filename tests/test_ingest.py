from types import SimpleNamespace

from stoic_rag.ingest import chunk_id, load_chunks


def _doc(text, page=0):
    return SimpleNamespace(page_content=text, metadata={"page": page})


def test_chunk_id_format():
    cid = chunk_id(_doc("hello", page=7))
    assert cid.startswith("p7-")
    assert len(cid) == len("p7-") + 16


def test_chunk_id_is_stable():
    assert chunk_id(_doc("same text")) == chunk_id(_doc("same text"))


def test_chunk_id_differs_per_content():
    assert chunk_id(_doc("a")) != chunk_id(_doc("b"))


def test_chunk_id_differs_per_page():
    assert chunk_id(_doc("a", page=1)) != chunk_id(_doc("a", page=2))


def test_chunk_id_defaults_page_to_zero():
    doc = SimpleNamespace(page_content="x", metadata={})
    assert chunk_id(doc).startswith("p0-")


def test_load_chunks_reports_missing_pdf(tmp_path):
    try:
        load_chunks(tmp_path / "absent.pdf", 1000, 100)
    except FileNotFoundError as exc:
        assert "Source PDF not found" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
