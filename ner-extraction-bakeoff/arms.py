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


async def spacy(text: str):
    from hermes.ner_provider import SpacyNERProvider
    from hermes.relation_extractor import SpacyRelationExtractor

    ner = SpacyNERProvider()
    re = SpacyRelationExtractor()
    entities = await ner.extract_entities(text)
    relations = await re.extract(text, entities)
    return _norm_entities(entities), _norm_relations(relations)


# Shared open NER+RE prompt. closed_vocab appends the reuse-vocab clause;
# big_model uses it unchanged on a larger model. Keeping one base prompt
# means closed_vocab vs the LLM arms differ only by the vocab clause / model.
_BASE_SYSTEM = (
    "You extract entities and relations from text for a knowledge graph.\n"
    "Return ONLY JSON: {\"entities\":[{\"name\":..,\"type\":..}],"
    "\"relations\":[{\"source\":..,\"relation\":..,\"target\":..}]}.\n"
    "source and target must be names from your entities list."
)
_VOCAB_CLAUSE = (
    "\nFor each relation, REUSE a relation from this known vocabulary when one "
    "fits; only coin a NEW relation label if none of these fit:\n{vocab}"
)


async def _llm_extract(text: str, system: str, model: str | None):
    """Run an open-prompt NER+RE extraction via generate_completion. The
    model is passed PER CALL (the only reliable override -- HERMES_LLM_MODEL
    is read at provider init, which is cached)."""
    from hermes.llm import generate_completion

    result = await generate_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        model=model,
        temperature=0.0,
        max_tokens=1024,
        metadata={"scenario": "bakeoff"},
    )
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        import re as _re

        m = _re.search(r"```(?:json)?\s*(.*?)```", content, _re.DOTALL)
        data = json.loads(m.group(1)) if m else {"entities": [], "relations": []}
    return _norm_entities(data.get("entities")), _norm_relations(data.get("relations"))


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
    """Same model as baseline (gpt-4o-mini), but the prompt injects the known
    relation vocabulary with reuse-don't-mint pressure."""
    vocab = ", ".join(relation_vocabulary())
    system = _BASE_SYSTEM + _VOCAB_CLAUSE.format(vocab=vocab)
    return await _llm_extract(text, system, model=None)


async def big_model(text: str):
    """Open prompt (no vocab constraint) on a larger model -- does a bigger
    model over-generate LESS, or is over-generation model-independent?"""
    model = os.environ.get("BAKEOFF_BIG_MODEL", "gpt-4o")
    return await _llm_extract(text, _BASE_SYSTEM, model=model)


ARMS = {
    "baseline": baseline,
    "spacy": spacy,
    "closed_vocab": closed_vocab,
    "big_model": big_model,
}
