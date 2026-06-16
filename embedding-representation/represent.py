"""The input representations under test.

Each takes a sample row {name, raw_text, start, end} and returns the string
that gets embedded. ``start``/``end`` are offsets into ``raw_text`` (end
exclusive: raw_text[start:end] == the mention surface form).

THE CONFOUND this set is built to avoid: ~15 entities are extracted from each
raw_text chunk (median 1084 chars), and 100% of entities share a chunk. A blind
char-window around the mention is therefore ~1/3 shared passage text, so
co-occurring entities collapse onto near-identical vectors -- you'd be measuring
chunk geometry, not entity geometry. So context is scoped to the *sentence*
containing the mention (cross-sentence entities separate), and the ``marked``
arm wraps the mention so even same-sentence entities differ.

  name           bare surface string (reproduces the live HCG baseline)
  sentence       the single sentence containing the mention
  name_sentence  "{name} — {sentence}" (identity anchored, context disambiguates)
  marked         sentence with the mention wrapped: "... «{name}» ..."
"""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"[.!?]\s+")


def sentence_span(rt: str, s: int, e: int) -> tuple[int, int]:
    """[a, b) of the sentence in rt that covers the mention [s, e)."""
    if not rt:
        return 0, 0
    starts = [0] + [m.end() for m in _BOUNDARY.finditer(rt)]
    ends = [m.start() + 1 for m in _BOUNDARY.finditer(rt)] + [len(rt)]
    for a, b in zip(starts, ends):
        if a <= s < b:
            return a, b
    return max(0, s - 80), min(len(rt), e + 80)  # fallback: tight window


def name_text(row: dict) -> str:
    return row["name"]


def sentence_text(row: dict) -> str:
    rt = row["raw_text"] or ""
    a, b = sentence_span(rt, int(row["start"]), int(row["end"]))
    return rt[a:b].strip() or row["name"]


def name_sentence_text(row: dict) -> str:
    return f"{row['name']} — {sentence_text(row)}"


def marked_text(row: dict) -> str:
    # locate the name within its sentence (case-insensitive) and wrap it -- more
    # robust than the stored offsets, which drift by ~1 char.
    sent = sentence_text(row)
    name = row["name"]
    i = sent.lower().find(name.lower())
    if i >= 0:
        return f"{sent[:i]}«{sent[i:i + len(name)]}»{sent[i + len(name):]}".strip()
    return f"«{name}» {sent}".strip()


def gloss_text(row: dict) -> str:
    """SENSE facet: the generated chunk-independent definition (attached by
    gloss.attach_glosses). Falls back to the bare name if no gloss is present."""
    return (row.get("gloss") or "").strip() or row["name"]


def name_gloss_text(row: dict) -> str:
    """Identity-anchored sense: '{name} — {gloss}'."""
    g = (row.get("gloss") or "").strip()
    return f"{row['name']} — {g}" if g else row["name"]


REPS = {
    "name": name_text,
    "sentence": sentence_text,
    "name_sentence": name_sentence_text,
    "marked": marked_text,
    "gloss": gloss_text,
    "name_gloss": name_gloss_text,
}
