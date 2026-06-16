"""SENSE facet: a chunk-independent gloss per entity, embedded instead of the
bare name.

One short definition per UNIQUE name, generated from the name + the sentence it
was mentioned in, but explicitly FORBIDDEN to quote or paraphrase that sentence.
The sentence only disambiguates the sense; the embedded output is a clean
definition carrying no passage vocabulary -- so unlike the Run-1 context arms it
should NOT cluster by source chunk.

    OPENAI_API_KEY=... .venv/bin/python gloss.py   # generate -> .cache/glosses.json
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
from pathlib import Path

import httpx

from represent import sentence_text

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CACHE.mkdir(exist_ok=True)
GLOSS_PATH = CACHE / "glosses.json"
GEN_MODEL = "gpt-4o-mini"

_PROMPT = (
    "Give a one-sentence, dictionary-style definition of the term below, as it "
    "is used in the example sentence. Write a self-contained definition of the "
    "concept ONLY. Do NOT quote or paraphrase the example sentence, do NOT "
    "mention the example, and do NOT name other entities from it.\n\n"
    "Term: {name}\n"
    "Example sentence (for disambiguation only): {sentence}\n\n"
    "Definition:"
)


def build_prompt(name: str, sentence: str) -> str:
    return _PROMPT.format(name=name, sentence=sentence)
