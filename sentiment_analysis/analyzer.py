"""Optimized lexicon-based sentiment analysis.

Requires spaCy and TextBlob.

Input lexicon CSV:
    word,polarity,intensity

Returns:
    lexicon_polarity
    textblob_polarity
    textblob_subjectivity
"""

from __future__ import annotations

import csv
from pathlib import Path

import spacy
from textblob import TextBlob


NEGATORS = {"not", "n't", "never", "no", "neither", "nor"}
OPINION_POS = {"ADJ", "VERB", "NOUN"}


def clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def load_lexicon(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load lexicon as word -> (polarity, intensity)."""

    entries = {}

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        required = {"word", "polarity", "intensity"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV requires columns: {', '.join(sorted(required))}"
            )

        for row_num, row in enumerate(reader, start=2):
            word = (row["word"] or "").strip().casefold()

            if not word:
                continue

            if word in entries:
                raise ValueError(
                    f"Duplicate lexicon word {word!r} at row {row_num}"
                )

            try:
                polarity = float(row["polarity"])
                intensity = float(row["intensity"])

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid scores at row {row_num}"
                ) from exc

            if not -1 <= polarity <= 1 or not 0 < intensity <= 10:
                raise ValueError(
                    f"Out-of-range scores at row {row_num}"
                )

            entries[word] = (polarity, intensity)

    return entries


class AspectSentimentAnalyzer:

    def __init__(self, lexicon_path: str | Path, model: str = "en_core_web_sm", negation_factor: float = -0.8):
        self.lexicon = load_lexicon(lexicon_path)
        self.negation_factor = negation_factor

        self.nlp = spacy.load(
            model,
            disable=["ner"]
        )

        if not self.nlp.has_pipe("parser"):
            raise ValueError(
                "spaCy model must include a dependency parser"
            )

    def entry(self, token) -> tuple[float, float] | None:
        """Return (polarity, intensity) for a token."""

        return (
            self.lexicon.get(token.lemma_.casefold())
            or self.lexicon.get(token.text.casefold())
        )

    def _negated(self, token) -> bool:
        """Detect negation on an opinion or its governing predicate."""

        nodes = [token]

        if token.dep_ in {"acomp", "attr", "oprd", "amod"}:
            nodes.append(token.head)

        for node in nodes:
            for child in node.children:

                if child.dep_ == "neg":
                    return True

                if (
                    child.lower_ in NEGATORS
                    and child.dep_ in {"advmod", "det"}
                ):
                    return True

        return False

    def _score_opinion(self, token) -> float | None:
        """Calculate a single opinion's sentiment score."""

        entry = self.entry(token)

        if entry is None:
            return None

        polarity, intensity = entry

        if polarity == 0:
            return None

        multiplier = intensity

        for child in token.children:

            if child.dep_ != "advmod":
                continue

            if child.lower_ in NEGATORS:
                continue

            modifier_entry = self.entry(child)

            if modifier_entry is not None:
                multiplier *= modifier_entry[1]

        if self._negated(token):
            multiplier *= self.negation_factor

        score = clamp(polarity * multiplier)

        return round(score, 4)

    def analyze(self, text: str) -> dict[str, float]:
        """Return only the three requested sentiment metrics."""

        doc = self.nlp(text)

        total_polarity = 0.0
        opinion_count = 0

        for token in doc:

            if token.is_punct:
                continue

            if token.pos_ not in OPINION_POS:
                continue

            score = self._score_opinion(token)

            if score is None:
                continue

            total_polarity += score
            opinion_count += 1

        lexicon_polarity = (
            round(total_polarity / opinion_count, 4)
            if opinion_count
            else 0.0
        )

        sentiment = TextBlob(text).sentiment

        return {
            "lexicon_polarity": lexicon_polarity,
            "textblob_polarity": round(sentiment.polarity, 4),
            "textblob_subjectivity": round(sentiment.subjectivity, 4)
        }


if __name__ == "__main__":

    analyzer = AspectSentimentAnalyzer(
        Path(__file__).with_name("sentiment_lexicon.csv")
    )

    text = "I absolutely love the animation, but I hate the story."

    print(analyzer.analyze(text))