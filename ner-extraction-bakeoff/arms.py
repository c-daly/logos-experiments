"""The 4 extraction arms for the bake-off (logos-experiments#38).

Each arm: async run(text) -> (entities, relations), normalized to
entity {name,type} and relation {source,relation,target}. Runs in the
hermes venv. The OpenAI arms call the real extractors / generate_completion;
the spaCy arm uses the local providers. closed_vocab is built here (a
vocab-injected prompt) so production code is untouched.
"""

from __future__ import annotations

import json
import os


def _norm_entities(raw: list[dict]) -> list[dict]:
    out = []
    for e in raw or []:
        name = e.get("name") or e.get("text")
        if name:
            out.append({"name": name, "type": e.get("type", "entity")})
    return out


def _norm_relations(raw: list[dict]) -> list[dict]:
    out = []
    for r in raw or []:
        src = r.get("source") or r.get("source_name")
        tgt = r.get("target") or r.get("target_name")
        rel = r.get("relation")
        if src and tgt and rel:
            out.append({"source": src, "relation": rel, "target": tgt})
    return out


async def _combined(text: str, model: str | None) -> tuple[list[dict], list[dict]]:
    from hermes.combined_extractor import OpenAICombinedExtractor

    ex = OpenAICombinedExtractor()
    if model:
        os.environ["HERMES_LLM_MODEL"] = model
    entities, relations = await ex.extract_entities_and_relations(text)
    return _norm_entities(entities), _norm_relations(relations)


async def baseline(text: str):
    return await _combined(text, model=None)


async def cheap_model(text: str):
    model = os.environ.get("BAKEOFF_CHEAP_MODEL", "gpt-4o-mini")
    return await _combined(text, model=model)


async def spacy(text: str):
    from hermes.ner_provider import SpacyNERProvider
    from hermes.relation_extractor import SpacyRelationExtractor

    ner = SpacyNERProvider()
    re = SpacyRelationExtractor()
    entities = await ner.extract_entities(text)
    relations = await re.extract(text, entities)
    return _norm_entities(entities), _norm_relations(relations)


_CLOSED_SYSTEM = (
    "You extract entities and relations from text for a knowledge graph.\n"
    "Return ONLY JSON: {\"entities\":[{\"name\":..,\"type\":..}],"
    "\"relations\":[{\"source\":..,\"relation\":..,\"target\":..}]}.\n"
    "For each relation, REUSE a relation from this known vocabulary when one "
    "fits; only coin a NEW relation label if none of these fit:\n{vocab}\n"
    "source and target must be names from your entities list."
)


def relation_vocabulary(limit: int = 120) -> list[str]:
    """Known descriptive relations to inject. Prefer the Redis snapshot
    (logos:ontology:relations); fall back to a small seed."""
    try:
        import redis

        raw = redis.Redis(decode_responses=True).get("logos:ontology:relations")
        if raw:
            vocab = sorted(json.loads(raw).keys())
            return vocab[:limit]
    except Exception:
        pass
    return [
        "IS_A", "PART_OF", "LOCATED_IN", "PRODUCES", "USED_FOR", "EATS",
        "CATCHES", "AFFECTS", "MEMBER_OF", "PLAYS", "TOWS", "FASTER_THAN",
    ]


async def closed_vocab(text: str):
    from hermes.llm import generate_completion

    vocab = ", ".join(relation_vocabulary())
    result = await generate_completion(
        messages=[
            {"role": "system", "content": _CLOSED_SYSTEM.format(vocab=vocab)},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=1024,
        metadata={"scenario": "bakeoff_closed_vocab"},
    )
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        import re as _re

        m = _re.search(r"```(?:json)?\s*(.*?)```", content, _re.DOTALL)
        data = json.loads(m.group(1)) if m else {"entities": [], "relations": []}
    return _norm_entities(data.get("entities")), _norm_relations(data.get("relations"))


ARMS = {
    "baseline": baseline,
    "spacy": spacy,
    "closed_vocab": closed_vocab,
    "cheap_model": cheap_model,
}
