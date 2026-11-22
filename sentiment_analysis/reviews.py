import ast
import csv
import json
import re

from dataclasses import dataclass, field
from pathlib import Path

from textblob import TextBlob
from tqdm.auto import tqdm

from analyzer import AspectSentimentAnalyzer, OPINION_POS


@dataclass
class Review:
    uid: int
    profile: str
    anime_uid: int
    text: str
    score: int
    scores: dict[str, str]
    link: str
    processed_text: str
    cleaned_text: str = field(init=False)

    def __post_init__(self):
        self.cleaned_text = self.clean_text(self.text)

    @property
    def key(self):
        return f"{self.uid}, {self.anime_uid}"

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text or '').replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'(?im)^\s*(?:more\s+pics|helpful(?:\s*\(\d+\))?)\s*$', '', text)
        text = re.sub(r'(?m)^\s*[-=_]{3,}\s*$', '', text)

        lines = text.split('\n')
        pos = 0
        count = 0

        labels = re.compile(r'(?:Overall|Story|Animation|Sound|Character|Enjoyment)', re.I)

        while pos < len(lines):
            if not lines[pos].strip():
                pos += 1
                continue

            if labels.fullmatch(lines[pos].strip()):
                j = pos + 1

                while j < len(lines) and not lines[j].strip():
                    j += 1

                if j < len(lines) and re.fullmatch(r'10|[0-9]', lines[j].strip()):
                    count += 1
                    pos = j + 1
                    continue

            break

        if count:
            lines = lines[pos:]

        text = '\n'.join(lines)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)

        return re.sub(r'\n{3,}', '\n\n', text).strip()

    @classmethod
    def from_row(cls, row):
        raw = row.get('scores') or ''

        scores = (
            raw if isinstance(raw, dict)
            else ast.literal_eval(raw) if raw.strip()
            else {}
        )

        if not isinstance(scores, dict):
            raise ValueError('scores must be a dictionary')

        return cls(
            int(row['uid']),
            str(row.get('profile') or ''),
            int(row['anime_uid']),
            str(row.get('text') or ''),
            int(row['score']),
            {str(k): str(v) for k, v in scores.items()},
            str(row.get('link') or ''),
            str(row.get('processed_text') or '')
        )

    @classmethod
    def from_csv(cls, path):
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                yield cls.from_row(row)


class ReviewAnalyzer:
    def __init__(self, lexicon_path, model='en_core_web_sm'):
        self.analyzer = AspectSentimentAnalyzer(lexicon_path, model=model)
        self.results = {}

    def analyze_doc(self, doc):
        """Calculate sentiment from an already processed spaCy document."""

        total_polarity = 0.0
        opinion_count = 0

        for token in doc:
            if token.is_punct or token.pos_ not in OPINION_POS:
                continue

            score = self.analyzer._score_opinion(token)

            if score is not None:
                total_polarity += score
                opinion_count += 1

        lexicon_polarity = (
            round(total_polarity / opinion_count, 4)
            if opinion_count
            else 0.0
        )

        sentiment = TextBlob(doc.text).sentiment

        return {
            "lexicon_polarity": lexicon_polarity,
            "textblob_polarity": round(sentiment.polarity, 4),
            "textblob_subjectivity": round(sentiment.subjectivity, 4)
        }

    def analyze(self, review: Review):
        """Analyze one review without batch processing."""

        analysis = self.analyzer.analyze(review.cleaned_text)
        self.results[review.key] = analysis

        return analysis

    def analyze_csv(self, path, batch_size=1000, spacy_batch_size=32, output_dir=None):
        """Stream reviews through spaCy and save independent JSON batches."""

        reviews = Review.from_csv(path)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        def review_stream():
            for review in reviews:
                yield review.cleaned_text, review.key

        processed = self.analyzer.nlp.pipe(
            review_stream(),
            as_tuples=True,
            batch_size=spacy_batch_size,
            n_process=1
        )

        batch_number = 1
        item_count = 0

        for doc, key in tqdm(processed, desc="Analyzing reviews"):
            self.results[key] = self.analyze_doc(doc)
            item_count += 1

            if output_dir is not None and item_count >= batch_size:
                self.save_json(output_dir / f'batch_{batch_number:05d}.json')

                self.results.clear()
                item_count = 0
                batch_number += 1

        if output_dir is not None and item_count > 0:
            self.save_json(output_dir / f'batch_{batch_number:05d}.json')
            self.results.clear()

        return self.results

    def save_json(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent

    processor = ReviewAnalyzer(root / 'sentiment_lexicon.csv')

    processor.analyze_csv(
        path=root / 'clean_data' / 'reviews_clean.csv',
        batch_size=1000,
        spacy_batch_size=32,
        output_dir=root / 'analysis_batches'
    )

    print("Analysis complete.")