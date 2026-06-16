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


def _chat(prompt: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (needed to generate glosses)")
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": GEN_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _save(glosses: dict) -> None:
    GLOSS_PATH.write_text(json.dumps(glosses, indent=2, ensure_ascii=False))


def load_glosses() -> dict[str, str]:
    return json.loads(GLOSS_PATH.read_text()) if GLOSS_PATH.exists() else {}


def attach_glosses(sample: list[dict], glosses: dict[str, str]) -> None:
    """Set row['gloss'] in place from the name->gloss map."""
    for row in sample:
        row["gloss"] = glosses.get(row["name"], "")


def generate(sample: list[dict], workers: int = 16, save_every: int = 200) -> dict[str, str]:
    """One gloss per UNIQUE name (names are ~unique on this corpus: 1.04
    mentions/name). Parallel across names -- ~5k serial calls would take ~40 min;
    a thread pool makes it minutes. Cached + checkpointed so re-runs are free and
    a crash only loses the last <=save_every calls."""
    by_name: dict[str, str] = {}
    for row in sample:
        by_name.setdefault(row["name"], sentence_text(row))

    glosses = load_glosses()
    todo = [n for n in by_name if n not in glosses]
    if not todo:
        print(f"all {len(by_name)} glosses cached")
        return glosses
    print(f"generating {len(todo)} glosses ({len(by_name) - len(todo)} cached) ...")

    lock = threading.Lock()
    done = 0

    def work(name: str):
        try:
            return name, _chat(build_prompt(name, by_name[name]))
        except Exception as exc:  # leave it uncached -> retried on re-run
            print(f"  ! {name!r}: {exc}")
            return name, None

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for name, gloss in ex.map(work, todo):
            if gloss is None:
                continue
            with lock:
                glosses[name] = gloss
                done += 1
                if done % save_every == 0:
                    _save(glosses)
                    print(f"  {done}/{len(todo)} glossed")
    _save(glosses)
    print(f"wrote {GLOSS_PATH} ({len(glosses)} glosses)")
    return glosses


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    generate(sample)


if __name__ == "__main__":
    main()
