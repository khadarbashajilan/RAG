from stoic_rag.config import OUT_OF_SCOPE
from stoic_rag.query import clean_query, heuristic_cleanup, is_out_of_scope


def test_typos_expanded():
    # NB: "w" is not in the typo map, so the README's claimed
    # "w fwar" -> "with fear" rewrite does not actually happen.
    assert heuristic_cleanup("how do i dweal w fwar?") == "how do i deal w fear?"


def test_filler_removed():
    assert "um" not in heuristic_cleanup("um i basically really worry")


def test_lowercased_and_trimmed():
    assert heuristic_cleanup("  ANXIETY,  ") == "anxiety"


def test_whitespace_collapsed():
    assert heuristic_cleanup("a    b") == "a b"


def test_out_of_scope_detected():
    assert is_out_of_scope("best pizza in town")
    assert clean_query("what is the bitcoin price") == OUT_OF_SCOPE


def test_in_scope_not_flagged():
    assert clean_query("how do i deal with fear?") != OUT_OF_SCOPE


def test_stoic_question_survives_cleanup():
    assert clean_query("how do i dweal w fwar?") == "how do i deal w fear?"


def test_mapped_typos_fully_expand():
    assert heuristic_cleanup("abt fear, wut 2moro") == "about fear, what tomorrow"
