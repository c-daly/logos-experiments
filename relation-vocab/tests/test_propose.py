"""Unit tests for the W0.3b mapping proposer (logos-experiments#34).

Pure functions on synthetic vocabularies; the live mapping.csv run is the
measurement, these verify the machinery. Folding is a deliberately crude
stemmer -- proposals are human-reviewed, so recall beats precision -- but
it must be deterministic and never fold guarded forms (short tokens, -SS).
"""

import pytest

from embed_evidence import nearest_survivors
from propose import (
    Edge,
    Row,
    apply_embed_fallback,
    canon,
    content_tokens,
    fold_token,
    jaccard,
    propose_mappings,
)


class TestFolding:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ACTS", "ACT"),  # plural/3rd-person strip
            ("ADDRESSES", "ADDRESS"),  # -S strip then stem-E strip
            ("ADDRESSED", "ADDRESS"),  # -ED strip
            ("PRODUCES", "PRODUC"),  # meets PRODUCED/PRODUCING at the stem
            ("PRODUCED", "PRODUC"),
            ("PRODUCING", "PRODUC"),
            ("CARRIES", "CARRY"),  # -IES -> Y
            ("ASSESS", "ASSESS"),  # -SS guard: no strip
            ("IS", "IS"),  # short-token guard
            ("RED", "RED"),  # -ED guard on short token
            ("OF", "OF"),  # stopwords pass through fold untouched
        ],
    )
    def test_fold_token(self, raw, expected):
        assert fold_token(raw) == expected

    def test_canon_joins_folded_tokens(self):
        assert canon("ACTS_IN") == "ACT_IN"
        assert canon("ACT_IN") == "ACT_IN"

    def test_content_tokens_drop_prepositions_and_fold(self):
        assert content_tokens("ALIAS_FOR") == {"ALIAS"}
        assert content_tokens("AFFECTS_PURCHASING_POWER_OF") == {
            "AFFECT",
            "PURCHAS",
            "POWER",
        }


class TestJaccard:
    def test_overlap(self):
        assert jaccard({"A", "B"}, {"B", "C"}) == pytest.approx(1 / 3)

    def test_empty_sets_are_zero(self):
        assert jaccard(set(), {"A"}) == 0.0


def _edges():
    es = []
    # heads: LOCATED_IN (df=4, thing->location), ALIAS_OF (df=3)
    es += [Edge("LOCATED_IN", "thing", "location")] * 4
    es += [Edge("ALIAS_OF", "entity", "entity")] * 3
    # one-offs
    es.append(Edge("LOCATES_IN", "thing", "location"))  # canon collision -> LOCATED_IN? no: LOCATE_IN == LOCATE_IN after folds
    es.append(Edge("ALIAS_FOR", "entity", "entity"))  # token match -> ALIAS_OF
    es.append(Edge("SITUATED_IN", "thing", "location"))  # signature only
    es.append(Edge("QUUXED_FROBNICATED", "entity", "entity"))  # nothing
    return es


class TestProposals:
    def test_canon_collision_is_high_tier(self):
        rows = {r.predicate: r for r in propose_mappings(_edges())}
        r = rows["LOCATES_IN"]
        assert r.target == "LOCATED_IN" and r.tier == "high"
        assert "canon" in r.evidence

    def test_token_match_is_medium_tier(self):
        rows = {r.predicate: r for r in propose_mappings(_edges())}
        r = rows["ALIAS_FOR"]
        assert r.target == "ALIAS_OF" and r.tier == "medium"
        assert "token" in r.evidence

    def test_signature_only_is_low_tier(self):
        rows = {r.predicate: r for r in propose_mappings(_edges())}
        r = rows["SITUATED_IN"]
        assert r.target == "LOCATED_IN" and r.tier == "low"
        assert "signature" in r.evidence

    def test_no_evidence_is_keep(self):
        rows = {r.predicate: r for r in propose_mappings(_edges())}
        r = rows["QUUXED_FROBNICATED"]
        assert r.target == "" and r.tier == "keep"

    def test_polarity_marker_never_auto_maps(self):
        es = [Edge("REFERS_TO", "a", "b")] * 3 + [Edge("DOES_NOT_REFER_TO", "a", "b")]
        rows = {r.predicate: r for r in propose_mappings(es)}
        r = rows["DOES_NOT_REFER_TO"]
        assert r.target == "" and r.tier == "keep" and "polarity" in r.evidence

    def test_canon_group_of_one_offs_consolidates_to_first(self):
        es = [
            Edge("HEAD_REL", "a", "b"),
            Edge("HEAD_REL", "a", "b"),
            Edge("HEAD_REL", "a", "b"),
            Edge("ACT_IN", "person", "film"),
            Edge("ACTS_IN", "person", "film"),
        ]
        rows = {r.predicate: r for r in propose_mappings(es)}
        assert rows["ACTS_IN"].target == "ACT_IN"
        assert rows["ACTS_IN"].tier == "high"
        # the group's lexicographically-first member keeps no self-mapping
        assert rows["ACT_IN"].target != "ACT_IN"

    def test_only_one_offs_get_rows(self):
        preds = {r.predicate for r in propose_mappings(_edges())}
        assert "LOCATED_IN" not in preds and "ALIAS_OF" not in preds

    def test_rows_are_deterministic(self):
        a = [tuple(vars(r).values()) for r in propose_mappings(_edges())]
        b = [
            tuple(vars(r).values())
            for r in propose_mappings(list(reversed(_edges())))
        ]
        assert a == b


class TestEmbedFallback:
    """The complementary name-embedding pass only touches evidence-less keeps."""

    def _keep(self, p):
        return Row(p, 1, "", "keep", "")

    def test_high_sim_keep_becomes_embed_proposal(self):
        rows = [self._keep("AFFILIATED_WITH")]
        apply_embed_fallback(rows, {"AFFILIATED_WITH": ("ASSOCIATED_WITH", 0.87)})
        r = rows[0]
        assert r.tier == "embed"
        assert r.target == "ASSOCIATED_WITH"
        assert "0.87" in r.evidence

    def test_low_sim_keep_stays_keep_but_records_neighbour(self):
        rows = [self._keep("ACCOMPANIED_BY")]
        apply_embed_fallback(rows, {"ACCOMPANIED_BY": ("ASSOCIATED_WITH", 0.59)})
        r = rows[0]
        assert r.tier == "keep"
        assert r.target == ""  # not a proposal, just evidence
        assert "nearest (kept)" in r.evidence and "0.59" in r.evidence

    def test_existing_evidence_is_untouched(self):
        # polarity keeps carry their own evidence and must not be overwritten
        rows = [Row("DOES_NOT_REFER_TO", 1, "", "keep", "polarity marker -- review")]
        apply_embed_fallback(rows, {"DOES_NOT_REFER_TO": ("REFERS_TO", 0.99)})
        assert rows[0].tier == "keep" and "polarity" in rows[0].evidence

    def test_fired_tier_is_untouched(self):
        rows = [Row("LOCATES_IN", 1, "LOCATED_IN", "high", "canon collision: LOCATE_IN")]
        apply_embed_fallback(rows, {"LOCATES_IN": ("SOMETHING_ELSE", 0.99)})
        assert rows[0].tier == "high" and rows[0].target == "LOCATED_IN"

    def test_keep_without_neighbour_stays_bare(self):
        rows = [self._keep("UTTERLY_UNIQUE")]
        apply_embed_fallback(rows, {})
        assert rows[0].tier == "keep" and rows[0].evidence == ""


class TestNearestSurvivors:
    def test_picks_highest_cosine_survivor(self):
        vectors = {
            "CAUSES": [1.0, 0.0],
            "TRIGGERS": [0.99, 0.14],  # closest to CAUSES
            "UNRELATED": [0.0, 1.0],
        }
        out = nearest_survivors(["TRIGGERS"], {"CAUSES", "UNRELATED"}, vectors)
        nn, sim = out["TRIGGERS"]
        assert nn == "CAUSES" and sim > 0.95

    def test_never_maps_a_predicate_to_itself(self):
        # a one-off that also (erroneously) appears among survivors
        vectors = {"DUP": [1.0, 0.0], "OTHER": [0.6, 0.8]}
        out = nearest_survivors(["DUP"], {"DUP", "OTHER"}, vectors)
        assert out["DUP"][0] == "OTHER"

    def test_one_off_without_vector_is_skipped(self):
        out = nearest_survivors(["NOVEC"], {"CAUSES"}, {"CAUSES": [1.0, 0.0]})
        assert "NOVEC" not in out
