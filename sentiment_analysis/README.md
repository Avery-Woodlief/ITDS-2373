# Dependency-aware sentiment analyzer

Combines **spaCy** dependency parsing with **TextBlob** polarity and subjectivity analysis and a **user-supplied CSV sentiment lexicon**.

The analyzer identifies sentiment-bearing words, accounts for modifiers and negation, and calculates three numerical values describing the overall sentiment of a text.

## Installation

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```python
from analyzer import AspectSentimentAnalyzer

analyzer = AspectSentimentAnalyzer("sentiment_lexicon.csv")

result = analyzer.analyze("The movie was not very good.")

print(result["lexicon_polarity"])
print(result["textblob_polarity"])
print(result["textblob_subjectivity"])
```

## Lexicon format

UTF-8 CSV with required columns `word,polarity,intensity`.

- `word`: Surface form or lemma (case-insensitive).
- `polarity`: Sentiment value between -1 and +1.
- `intensity`: Positive multiplier (1 = neutral intensity, >1 intensifies, <1 weakens).
- Modifiers like `very` should usually have polarity 0 and intensity >1.
- Negators (`not`, `never`, `no`) are handled structurally rather than as negative-polarity words.
- Lexicon scores are heuristic and depend on the values assigned to individual words.

## Outputs

The analyzer returns three numerical values:

```json
{
    "lexicon_polarity": 0.0735,
    "textblob_polarity": 0.102,
    "textblob_subjectivity": 0.5356
}
```

### 1. Lexicon Polarity

`lexicon_polarity` measures the overall sentiment of the text using the custom sentiment lexicon.

Its numerical range is:

**-1.0 ≤ Polarity ≤ +1.0**

| Value | Interpretation |
|---|---|
| -1.0 | Extremely negative sentiment |
| -0.5 | Moderately negative sentiment |
| 0.0 | Neutral or balanced sentiment |
| +0.5 | Moderately positive sentiment |
| +1.0 | Extremely positive sentiment |

Negative values indicate negative sentiment, while positive values indicate positive sentiment.

A value close to zero indicates that the average detected sentiment is relatively neutral or that positive and negative opinions balance each other.

For example:

```json
"lexicon_polarity": 0.0735
```

This indicates **very slightly positive sentiment** according to the custom lexicon.

The value is calculated by averaging the individual opinion scores identified in the text.

### 2. TextBlob Polarity

`textblob_polarity` measures the overall sentiment of the text using TextBlob's predefined sentiment lexicon and analysis rules.

Its numerical range is:

**-1.0 ≤ Polarity ≤ +1.0**

| Value | Interpretation |
|---|---|
| -1.0 to -0.6 | Strongly negative |
| -0.6 to -0.2 | Moderately negative |
| -0.2 to 0.0 | Slightly negative |
| 0.0 | Neutral or balanced |
| 0.0 to +0.2 | Slightly positive |
| +0.2 to +0.6 | Moderately positive |
| +0.6 to +1.0 | Strongly positive |

These ranges are illustrative guidelines rather than official TextBlob classification thresholds.

For example:

```json
"textblob_polarity": 0.102
```

This indicates **slightly positive sentiment**.

Although both polarity measurements use the same numerical range, they are calculated independently and may produce different results.

### 3. TextBlob Subjectivity

`textblob_subjectivity` measures how subjective or opinion-based the language is.

Its numerical range is:

**0.0 ≤ Subjectivity ≤ 1.0**

| Value | Interpretation |
|---|---|
| 0.0 | Highly objective |
| 0.1–0.3 | Mostly objective |
| 0.3–0.5 | Somewhat objective |
| 0.5–0.7 | Moderately subjective |
| 0.7–0.9 | Highly subjective |
| 0.9–1.0 | Extremely subjective |

These categories are approximate interpretations rather than official thresholds.

**Objective language** primarily communicates factual information.

Example:

> "The anime contains 24 episodes."

**Subjective language** expresses opinions, feelings, preferences, or personal judgments.

Example:

> "The animation was absolutely incredible."

For example:

```json
"textblob_subjectivity": 0.5356
```

This indicates **moderately subjective language**.

A subjectivity score of 0.5356 does not mean that exactly 53.56% of the text is subjective. It represents a numerical estimate of subjectivity based on TextBlob's analysis.

### 4. Relationship Between Polarity and Subjectivity

Polarity and subjectivity measure different characteristics of a text.

- **Polarity** measures whether the expressed sentiment is negative, neutral, or positive.
- **Subjectivity** measures how strongly the language reflects opinions rather than factual statements.

A text can have high subjectivity while maintaining a polarity close to zero.

For example:

> "The animation was amazing, but the story was terrible."

This sentence contains strong positive and negative opinions that may balance each other, producing a polarity near zero despite its subjective language.

Therefore, a polarity close to zero does not necessarily indicate an absence of strong opinions.

### 5. Interpreting All Three Values Together

Consider the following output:

```json
{
    "lexicon_polarity": 0.0735,
    "textblob_polarity": 0.102,
    "textblob_subjectivity": 0.5356
}
```

Interpretation:

- **Lexicon polarity (0.0735):** The custom lexicon identifies very slightly positive sentiment.
- **TextBlob polarity (0.102):** TextBlob independently identifies slightly positive sentiment.
- **TextBlob subjectivity (0.5356):** The text contains moderately subjective language.

Overall, these results suggest that the text expresses **slightly positive sentiment with a moderate degree of subjectivity**.

The two polarity scores should not be added together because they are produced by different scoring methods.

## Scoring Rule

The custom lexicon calculates individual opinion scores using:

```text
score = clamp(
    base_polarity
    × base_intensity
    × product(modifier_intensities)
    × negation_factor,
    -1,
    1
)
```

The default negation factor is `-0.8`.

For example, consider an opinion with:

```text
base_polarity = 0.5
base_intensity = 1.0
modifier_intensity = 1.5
negation_factor = 1.0
```

The calculated sentiment is:

```text
score = 0.5 × 1.0 × 1.5 × 1.0
score = 0.75
```

The overall lexicon polarity is the arithmetic mean of the detected opinion scores:

```text
lexicon_polarity = sum(opinion_scores) / number_of_opinions
```

If no scored opinions are detected, the analyzer returns a lexicon polarity of `0.0`.

This is a heuristic scoring system rather than a calibrated statistical model.

## Limitations

This analyzer uses a **rule-based approach** rather than a trained sentiment classification model.

Its limitations include:

- Difficulty interpreting sarcasm and irony.
- Incomplete recognition of complex negation.
- Limited understanding of context-dependent sentiment.
- Potential inaccuracies when processing multiword expressions.
- Dependence on the coverage and accuracy of the custom lexicon.
- Positive and negative opinions potentially canceling each other out.
- Differences between the custom lexicon and TextBlob scoring methods.

A polarity value close to zero may indicate either genuinely neutral language or a mixture of positive and negative opinions.

Similarly, subjectivity measures the characteristics of the language rather than the factual correctness of its statements.

The three numerical values should therefore be interpreted as **estimates of sentiment and subjectivity**, not exact measurements or probabilities.