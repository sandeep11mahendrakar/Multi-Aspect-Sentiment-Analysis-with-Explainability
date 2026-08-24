"""Shared sentence-splitting + aspect-tagging utility.

Dependency-free (regex based - clothing reviews need nothing heavier).
Aspect taxonomy reuses the EXACT keyword sets from apps/app.py so E7B
training pairs stay consistent with the deployed dashboard.

Usage:
    from sentence_splitter import split_sentences, tag_aspects
"""

import re

ASPECT_KEYWORDS = {
    "quality": ["quality", "material", "fabric", "build", "construction", "durability"],
    "price": ["price", "cheap", "expensive", "cost", "value", "worth"],
    "fit": ["fit", "fits", "fitted", "fitting", "size", "sized", "sizing",
            "tight", "loose", "small", "large"],
    "shipping": ["delivery", "shipping", "delivered", "arrived", "package"],
}

_WORD_RE = re.compile(r"[a-z']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"']?\s+[\"']?(?=[A-Za-z\"'])")


def split_sentences(text: str):
    """Split a review into sentences; falls back to whole text if no delimiters."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def tag_aspects(sentence: str):
    """Return set of aspect names whose keywords appear in the sentence."""
    words = set(_WORD_RE.findall(sentence.lower()))
    return {aspect for aspect, kws in ASPECT_KEYWORDS.items()
            if words.intersection(kws)}


if __name__ == "__main__":
    demo = ("Love the fabric! The size runs small though. "
            "Shipping was fast, but a bit expensive for what it is.")
    for s in split_sentences(demo):
        print(f"{sorted(tag_aspects(s)) or ['(none)']}: {s}")
