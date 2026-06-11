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
import sys
from functools import lru_cache


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


async def _combined(text: str) -> tuple[list[dict], list[dict]]:
    """Baseline: the production combined extractor exactly as it ships.
    HERMES_LLM_MODEL is baked into the cached llm provider at first use, so a
    per-call model override would not take effect here -- baseline is the
    as-shipped extractor (default model) by design."""
    from hermes.combined_extractor import OpenAICombinedExtractor

    ex = OpenAICombinedExtractor()
    entities, relations = await ex.extract_entities_and_relations(text)
    return _norm_entities(entities), _norm_relations(relations)


async def baseline(text: str):
    return await _combined(text)


async def spacy(text: str):
    """spaCy NAMED-entity recognition (doc.ents) + dependency RE. NER only
    catches PERSON/ORG/GPE/DATE etc. -- a non-starter for common-noun domain
    entities. Kept to show that ceiling."""
    from hermes.ner_provider import SpacyNERProvider
    from hermes.relation_extractor import SpacyRelationExtractor

    ner = SpacyNERProvider()
    re = SpacyRelationExtractor()
    entities = await ner.extract_entities(text)
    relations = await re.extract(text, entities)
    return _norm_entities(entities), _norm_relations(relations)


_DETERMINERS = {
    "a", "an", "the", "its", "his", "her", "their", "our", "your", "my",
    "whose", "this", "that", "these", "those", "some", "any", "each",
}


def _chunk_entities(doc) -> list[dict]:
    """Noun-chunk entities with leading determiners/pronouns stripped."""
    entities = []
    for chunk in doc.noun_chunks:
        toks = [
            t.text for t in chunk
            if t.text.lower() not in _DETERMINERS and t.pos_ not in ("DET", "PRON")
        ]
        name = " ".join(toks).strip()
        if name:
            entities.append({"name": name, "type": "entity"})
    return entities


def _pos_entities(doc) -> list[dict]:
    """Single NOUN/PROPN tokens as entities (dedup by lowercased name)."""
    seen, out = set(), []
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and tok.text.lower() not in seen:
            seen.add(tok.text.lower())
            out.append({"name": tok.text, "type": "entity"})
    return out


def _head_entities(doc) -> list[dict]:
    """Just the head noun of each noun chunk (chunk.root) -- "a slow delivery
    truck" -> "truck". Dedup by lowercased name."""
    seen, out = set(), []
    for chunk in doc.noun_chunks:
        name = chunk.root.text
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "type": "entity"})
    return out


async def spacy_chunks(text: str):
    """spaCy NOUN-CHUNK entity extraction (the right primitive for common-noun
    domain entities NER misses) + dependency RE. Free/local; tests whether
    spaCy is viable for NODES even though its relations are weak."""
    from hermes.relation_extractor import SpacyRelationExtractor
    from hermes.services import get_spacy_model

    doc = get_spacy_model()(text)
    entities = _chunk_entities(doc)
    relations = await SpacyRelationExtractor().extract(text, entities)
    return _norm_entities(entities), _norm_relations(relations)


async def spacy_pos(text: str):
    """spaCy NOUN/PROPN-token entities (single-word node primitive) + dep RE.
    The other free node extractor, to contrast with noun-chunks."""
    from hermes.relation_extractor import SpacyRelationExtractor
    from hermes.services import get_spacy_model

    doc = get_spacy_model()(text)
    entities = _pos_entities(doc)
    relations = await SpacyRelationExtractor().extract(text, entities)
    return _norm_entities(entities), _norm_relations(relations)


async def spacy_head(text: str):
    """spaCy head-noun entities (just the noun, chunk.root) + dep RE -- the
    leanest free node primitive."""
    from hermes.relation_extractor import SpacyRelationExtractor
    from hermes.services import get_spacy_model

    doc = get_spacy_model()(text)
    entities = _head_entities(doc)
    relations = await SpacyRelationExtractor().extract(text, entities)
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
        try:
            data = json.loads(m.group(1)) if m else {}
        except json.JSONDecodeError:
            data = {}
    return _norm_entities(data.get("entities")), _norm_relations(data.get("relations"))


_SEED_RELATIONS: tuple[str, ...] = (
    "IS_A", "PART_OF", "LOCATED_IN", "PRODUCES", "USED_FOR", "EATS",
    "CATCHES", "AFFECTS", "MEMBER_OF", "PLAYS", "TOWS", "FASTER_THAN",
)


@lru_cache(maxsize=1)
def relation_vocabulary(limit: int = 120) -> tuple[str, ...]:
    """Known descriptive relations to inject. Prefer the Redis snapshot
    (logos:ontology:relations); fall back to a small seed. Cached (maxsize=1):
    the snapshot is fetched once per process, not once per sentence. Returns a
    tuple so the cached value stays immutable. Falling back to the seed is
    announced on stderr -- it is a much smaller (stronger) vocab than a live
    run, so closed_vocab metrics would otherwise differ silently."""
    try:
        import redis

        client = redis.Redis(decode_responses=True)
        try:
            raw = client.get("logos:ontology:relations")
        finally:
            client.close()
        if raw:
            return tuple(sorted(json.loads(raw).keys())[:limit])
        reason = "snapshot key 'logos:ontology:relations' is empty/absent"
    except Exception as exc:
        reason = f"Redis unavailable ({exc})"
    print(
        f"  [relation_vocabulary] {reason}; falling back to the "
        f"{len(_SEED_RELATIONS)}-relation seed -- closed_vocab metrics will "
        "differ from a live-Redis run",
        file=sys.stderr,
    )
    return _SEED_RELATIONS


# A clean, compact descriptive-relation vocabulary: the 20 gold relations
# plus common others, so the model must CHOOSE (not echo the answers). This
# is the production analog of a curated / rolled-up relation set.
_CLEAN_VOCAB = [
    "IS_A", "PART_OF", "HAS_PART", "LOCATED_IN", "PRODUCES", "USED_FOR",
    "EATS", "CATCHES", "AFFECTS", "MEMBER_OF", "PLAYS", "TOWS",
    "FASTER_THAN", "BEHIND", "TUNES", "ORIENTED_TOWARD", "POWERED_BY",
    "CARRIES", "MADE_OF", "CONTAINS", "ORBITS", "CAUSES", "USES",
    "FOUND_IN", "DERIVED_FROM", "ASSOCIATED_WITH", "HAS_PROPERTY",
    "OCCURS_IN", "COMPOSED_OF", "INCLUDES", "SIMILAR_TO", "ADJACENT_TO",
]


async def _closed_vocab(text: str, vocab_list: list[str]):
    system = _BASE_SYSTEM + _VOCAB_CLAUSE.format(vocab=", ".join(vocab_list))
    return await _llm_extract(text, system, model=None)


async def closed_vocab(text: str):
    """gpt-4o-mini + the LIVE relation vocabulary injected (reuse-don't-mint).
    The live snapshot is the already-sprawled ~2,300-relation set."""
    return await _closed_vocab(text, relation_vocabulary())


async def closed_vocab_clean(text: str):
    """gpt-4o-mini + a CLEAN compact (~32) relation vocabulary injected --
    the production analog of a curated/rolled-up set."""
    return await _closed_vocab(text, _CLEAN_VOCAB)


def _ranked(snapshot: dict, limit: int) -> tuple[str, ...]:
    """Top *limit* relation names by edge_count desc, name tiebreak -- the
    post-fix H5 window strategy (hermes feat/hermes-h5-ranked-vocab-window),
    vs relation_vocabulary()'s alphabetical slice."""

    def count(props) -> int:
        return props.get("edge_count", 0) if isinstance(props, dict) else 0

    ranked = sorted(snapshot.items(), key=lambda kv: (-count(kv[1]), kv[0]))
    return tuple(name for name, _ in ranked[:limit])


@lru_cache(maxsize=1)
def relation_vocabulary_ranked(limit: int = 120) -> tuple[str, ...]:
    """LIVE snapshot, usage-ranked window. Same fetch/fallback contract as
    relation_vocabulary(); differs ONLY in how the slice is chosen."""
    try:
        import redis

        client = redis.Redis(decode_responses=True)
        try:
            raw = client.get("logos:ontology:relations")
        finally:
            client.close()
        if raw:
            return _ranked(json.loads(raw), limit)
        reason = "snapshot key 'logos:ontology:relations' is empty/absent"
    except Exception as exc:
        reason = f"Redis unavailable ({exc})"
    print(
        f"  [relation_vocabulary_ranked] {reason}; falling back to the "
        f"{len(_SEED_RELATIONS)}-relation seed",
        file=sys.stderr,
    )
    return _SEED_RELATIONS


@lru_cache(maxsize=1)
def relation_vocabulary_curated(limit: int = 120) -> tuple[str, ...]:
    """Usage-ranked window over the consolidation survivors
    (relation-vocab/curated_vocab.json) -- the curated-seed + ranked-window
    pipeline."""
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "relation-vocab"
        / "curated_vocab.json"
    )
    try:
        return _ranked(json.loads(path.read_text()), limit)
    except Exception as exc:
        print(
            f"  [relation_vocabulary_curated] {path} unavailable ({exc}); "
            f"falling back to the {len(_SEED_RELATIONS)}-relation seed",
            file=sys.stderr,
        )
        return _SEED_RELATIONS


async def ranked_window(text: str):
    """gpt-4o-mini + LIVE vocab, top-120 by edge_count -- production's
    post-fix H5 window strategy over the same sprawled snapshot that
    closed_vocab slices alphabetically."""
    return await _closed_vocab(text, list(relation_vocabulary_ranked()))


async def ranked_window_curated(text: str):
    """gpt-4o-mini + consolidation-survivor vocab, top-120 by count -- the
    full intended pipeline (curated seed + ranked window)."""
    return await _closed_vocab(text, list(relation_vocabulary_curated()))


async def big_model(text: str):
    """Open prompt (no vocab constraint) on a larger model -- does a bigger
    model over-generate LESS, or is over-generation model-independent?"""
    model = os.environ.get("BAKEOFF_BIG_MODEL", "gpt-4o")
    return await _llm_extract(text, _BASE_SYSTEM, model=model)


async def big_model_clean(text: str):
    """Larger model + clean compact vocab -- both levers at once."""
    model = os.environ.get("BAKEOFF_BIG_MODEL", "gpt-4o")
    system = _BASE_SYSTEM + _VOCAB_CLAUSE.format(vocab=", ".join(_CLEAN_VOCAB))
    return await _llm_extract(text, system, model=model)


# NB: contains literal JSON braces, so it must NOT be .format()'d directly --
# only _VOCAB_CLAUSE (which has the {vocab} field) is formatted, then appended.
_REL_ONLY_SYSTEM = (
    "You are given a text and a list of entities. Extract the relations "
    "BETWEEN those entities. Return ONLY JSON: "
    "{\"relations\":[{\"source\":..,\"relation\":..,\"target\":..}]}. "
    "source and target MUST be exact names from the given entities list."
)


async def hybrid_clean(text: str):
    """The cost-optimal combo: FREE spaCy noun-chunk entities + a CHEAP LLM
    (gpt-4o-mini) relations-only pass constrained to the clean vocab."""
    from hermes.llm import generate_completion
    from hermes.services import get_spacy_model

    entities = _chunk_entities(get_spacy_model()(text))
    if len(entities) < 2:
        return _norm_entities(entities), []
    names = [e["name"] for e in entities]
    system = _REL_ONLY_SYSTEM + _VOCAB_CLAUSE.format(vocab=", ".join(_CLEAN_VOCAB))
    result = await generate_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Entities: {names}\nText: {text}"},
        ],
        model=None,
        temperature=0.0,
        max_tokens=1024,
        metadata={"scenario": "bakeoff_hybrid"},
    )
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        import re as _re

        m = _re.search(r"```(?:json)?\s*(.*?)```", content, _re.DOTALL)
        data = json.loads(m.group(1)) if m else {"relations": []}
    return _norm_entities(entities), _norm_relations(data.get("relations"))


ARMS = {
    # LLM extraction (entities + relations in one call)
    "baseline": baseline,                    # gpt-4o-mini, open prompt (production)
    "closed_vocab": closed_vocab,            # mini + live (sprawled) vocab
    "closed_vocab_clean": closed_vocab_clean,  # mini + clean compact vocab
    "ranked_window": ranked_window,          # mini + live vocab, top-120 by edge_count (H5 post-fix)
    "ranked_window_curated": ranked_window_curated,  # mini + curated survivors, top-120 by count
    "big_model": big_model,                  # gpt-4o, open prompt
    "big_model_clean": big_model_clean,      # gpt-4o + clean vocab
    # spaCy node extractors (free/local) + dependency RE
    "spacy": spacy,                          # NER (doc.ents) -- the ceiling
    "spacy_pos": spacy_pos,                  # NOUN/PROPN tokens
    "spacy_chunks": spacy_chunks,            # full noun chunks (det-stripped)
    "spacy_head": spacy_head,                # head noun only
    # cost-optimal hybrid
    "hybrid_clean": hybrid_clean,            # spaCy chunk nodes + cheap LLM clean-vocab relations
}
