from represent import REPS, gloss_text, name_gloss_text


def test_gloss_text_uses_gloss_when_present():
    row = {"name": "pneumococcus", "gloss": "a bacterium that causes pneumonia"}
    assert gloss_text(row) == "a bacterium that causes pneumonia"


def test_gloss_text_falls_back_to_name_when_missing():
    assert gloss_text({"name": "pneumococcus"}) == "pneumococcus"
    assert gloss_text({"name": "pneumococcus", "gloss": ""}) == "pneumococcus"


def test_name_gloss_concatenates_with_em_dash():
    row = {"name": "pneumococcus", "gloss": "a bacterium that causes pneumonia"}
    assert name_gloss_text(row) == "pneumococcus — a bacterium that causes pneumonia"


def test_name_gloss_falls_back_to_bare_name():
    assert name_gloss_text({"name": "pneumococcus"}) == "pneumococcus"


def test_arms_registered():
    assert "gloss" in REPS and "name_gloss" in REPS
