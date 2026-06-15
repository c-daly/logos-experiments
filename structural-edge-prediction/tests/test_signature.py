"""signature: a sharp type concentrates mass; entropy ranks sharpness right."""

from __future__ import annotations

from harness.signature import (
    Signature,
    build_signatures,
    signature_entropy,
    signature_sharpness,
)


def test_whale_signature_gives_blowhole_high_gills_zero(toy_snapshot):
    sigs = build_signatures(toy_snapshot.edges, toy_snapshot)
    whale = sigs["whale"]
    # has->whale_anatomy (blowhole/tusk) carries real mass.
    assert whale.prob(("has", "whale_anatomy")) > 0.2
    # whales never source has->fish_anatomy (gills) -- exactly zero.
    assert whale.prob(("has", "fish_anatomy")) == 0.0
    # The discriminating trait edge is present too.
    assert whale.prob(("attr", "trait")) > 0.2


def test_fish_signature_mirrors_whale(toy_snapshot):
    sigs = build_signatures(toy_snapshot.edges, toy_snapshot)
    fish = sigs["fish"]
    assert fish.prob(("has", "fish_anatomy")) > 0.2
    assert fish.prob(("has", "whale_anatomy")) == 0.0


def test_sharp_type_has_lower_entropy_than_mixed_type():
    # A deliberately sharp signature: all mass on one pattern.
    sharp = Signature("sharp", {("has", "whale_anatomy"): 10})
    # A deliberately mixed signature: mass spread over four patterns.
    mixed = Signature(
        "mixed",
        {
            ("has", "whale_anatomy"): 3,
            ("lives_in", "habitat"): 3,
            ("attr", "trait"): 2,
            ("eats", "animal"): 2,
        },
    )
    assert signature_entropy(sharp) < signature_entropy(mixed)
    # Sharpness is the negation -- sharper => higher.
    assert signature_sharpness(sharp) > signature_sharpness(mixed)
    # The single-pattern signature has exactly zero entropy.
    assert signature_entropy(sharp) == 0.0


def test_zero_support_signature_is_inert():
    empty = Signature("empty", {})
    assert empty.support() == 0
    assert signature_entropy(empty) == 0.0
    assert signature_sharpness(empty) == float("-inf")
    assert empty.prob(("has", "whale_anatomy")) == 0.0
