# Anime Data Exploration and Sentiment Analysis

This project uses a Jupyter notebook (`exploring_data(1).ipynb`) to explore anime scores, popularity, genres, episode counts, review activity, and review sentiment. The visualizations help compare anime characteristics and identify patterns in audience ratings and review language.

## Setup

Install the Python packages used by the notebook:

```bash
pip install jupyter pandas numpy matplotlib seaborn
```

Run the notebook in Jupyter and keep the required data files in its working directory:

| File | Purpose |
| --- | --- |
| `anime_clean.csv` | Anime metadata, including scores, genres, episodes, popularity, rankings, and member counts. |
| `anime_review_counts.csv` | Review totals associated with anime IDs. |
| `sentiment_results.json` | Review-level Lexicon polarity, TextBlob polarity, and TextBlob subjectivity. |

Not every function uses every file. The notebook also uses Python standard-library modules such as `ast`, `json`, and `pathlib`.

## Visualization Functions

### Anime scores and characteristics

#### `Scores_of_Single_Episode_Anime()`
Plots score frequencies for anime with exactly one episode. Taller bars mean more single-episode anime received the corresponding score.

#### `Scores_of_Multi_Episode_Anime()`
Plots score frequencies for anime with more than one episode. Compare its distribution with the single-episode chart to see how score frequencies differ.

#### `Anime_Feature_Correlations()`
Displays a correlation-matrix heatmap for numerical anime features such as episodes, members, popularity, rank, and score. Values closer to **+1** or **−1** indicate stronger positive or negative linear relationships; values near **0** indicate weak linear relationships.

#### `Distribution_of_Anime_Scores_by_Genre(min_samples=30)`
Creates boxplots of anime scores by genre, excluding genres with fewer than `min_samples` observations. Compare medians, spread, and potential outliers; anime may appear in multiple genres.

#### `Distribution_of_Anime_Ranks_by_Genre(min_samples=30)`
Creates boxplots of anime ranking positions by genre. Lower numerical ranks represent better rankings; compare genre medians and variability.

#### `Distribution_of_Anime_Scores()`
Shows a histogram and smoothed density curve of anime scores. Taller bins indicate score ranges containing more anime.

#### `Anime_Genre_Frequency()`
Charts how frequently each genre occurs in the dataset. Taller bars represent more anime tagged with that genre; an anime can count toward multiple genres.

### Review counts and sentiment

#### `Number_of_Reviews_per_Anime_uid()`
Plots review count against anime ID. Higher points represent anime with more reviews. IDs are identifiers, so horizontal distance between points does not imply a meaningful numerical relationship.

#### `Top_30_Most_Reviewed_Anime()`
Displays the 30 anime with the highest review counts. Bar lengths represent review activity, **not** average ratings or sentiment.

#### `Anime_Sentiment_Top_30_Polarity()`
Compares mean Lexicon and TextBlob polarity for the 30 most-reviewed anime. Positive values indicate more positive language; negative values indicate more negative language. **Notebook note:** this function references `sentiment_df` without creating it locally, so that DataFrame must be defined before calling the function.

#### `Anime_Subjectivity_Top_30()`
Shows average TextBlob subjectivity for the 30 most-reviewed anime. Scores near **0** indicate more objective language; scores near **1** indicate more subjective or opinionated language.

#### `Anime_Subjectivity_vs_Polarity_Top_30()`
Compares subjectivity with Lexicon and TextBlob polarity for the 30 most-reviewed anime. Subjectivity measures opinionatedness (0–1), while polarity measures positive or negative sentiment (−1 to +1).

#### `Anime_Mirrored_Bar_Chart()`
Uses a mirrored chart to compare subjectivity and polarity magnitude. Subjectivity extends left; polarity magnitude extends right. Polarity bar **color**, rather than direction, indicates whether sentiment is positive or negative.

#### `Anime_Genre_Polarity_Top_30()`
Compares average Lexicon and TextBlob polarity across 30 genres ranked using their combined mean polarity. Rightward bars indicate positive sentiment and leftward bars negative sentiment. Reviews can contribute to multiple genres when their anime have multiple tags.

### Relationship analysis

#### `Rating_vs_Popularity()`
Creates scatter plots relating anime score to popularity rank and audience size (`members`). A lower popularity-rank number means greater popularity; trend lines summarize the overall relationships.

#### `Episode_Count_vs_Ratings()`
Plots episode count against anime score with a trend line and logarithmic episode axis. An upward trend suggests longer anime tend to have higher scores, but does not establish causation.

#### `Popularity_vs_Review_Volume()`
Compares review counts with popularity rank and member counts. Higher review counts indicate greater review activity, not necessarily more favorable opinions.

#### `Review_Sentiment_vs_Ratings()`
Compares anime scores with their average review polarity, showing Lexicon and TextBlob separately. Upward trends suggest higher-rated anime also tend to have more positive reviews; differences between methods highlight sentiment-model disagreement.

## Interpreting the Results

- **Ratings** are audience score measurements; **popularity rank** is ordered so smaller values indicate greater popularity.
- **Review volume** measures the number of reviews, not review positivity.
- **Polarity** measures sentiment direction, generally from −1 (negative) to +1 (positive).
- **Subjectivity** measures how opinion-based language is, generally from 0 (objective) to 1 (subjective).
- **Correlation** describes association, not causation. Regression lines and correlations should not be interpreted as proof that one feature causes another.
- **Genre comparisons** are overlapping: a single anime may belong to multiple genres.

## Project Scope

The notebook includes histograms, boxplots, genre-frequency bar charts, scatter plots, and a correlation-matrix heatmap. It supports exploratory analysis of rating versus popularity, genre versus scores, episode count versus ratings, popularity versus review volume, and review sentiment versus ratings.
