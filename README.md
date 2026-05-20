# 🎬 Movie Review Sentiment Analysis: Text Preprocessing & Vectorization 📊

![NLP Badge](https://img.shields.io/badge/NLP-Text_Processing-blueviolet)
![Python Badge](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn Badge](https://img.shields.io/badge/Scikit--learn-Vectorization-orange)

--- 

## 🌟 Project Overview

This  Notebook presents a comprehensive pipeline for preparing raw movie review text data for downstream Natural Language Processing (NLP) tasks, particularly sentiment analysis. The journey begins with raw text file having 50K reviews of movies and transforms it into various numerical representations, ready for machine learning models. 

I cover (Al hamd u lillah):
-   **Data Loading & Initial Exploration** 📥
-   **Robust Text Preprocessing** ✨
    -   Lowercasing, HTML/URL/Punctuation Removal
    -   Text Normalization (Repeating Chars, Whitespace)
    -   Spelling Correction with `SymSpellPy` ✍️
    -   Stopword Removal, Tokenization, and Lemmatization with `SpaCy` 🧠
-   **Data Checkpointing** 💾
-   **Advanced Text Vectorization Techniques** 📈
    -   One-Hot Encoding (OHE)
    -   Bag of Words (BoW) - Unigrams, Bigrams, Trigrams
    -   TF-IDF (Term Frequency-Inverse Document Frequency)
-   **Efficient Storage of Sparse Matrices** 🗜️

---

## 🚀 Getting Started

### 📦 Installation

To run this notebook, you'll need the following libraries. They are typically installed within the notebook itself, but you can also install them manually:

```bash
pip install pandas scikit-learn spacy symspellpy polars tqdm
python -m spacy download en_core_web_sm
```

### 📁 Data

The project uses the `IMDBDataset.csv` file, which should be placed in the `/content/dataset/` directory (or modify the paths in the notebook accordingly).

---

## 📝 Notebook Structure & Key Steps

Each major step in the notebook is clearly delineated with markdown headings and step numbers.

### 1. Data Loading & Initial Setup (Steps 1-3) 📥
-   **Load Dataset:** Reads `IMDBDataset.csv` into a pandas DataFrame, handling parsing errors with `engine='python'` and `on_bad_lines='skip'`. 
-   **Polars Integration:** An experimental section for loading data with `Polars` for performance comparison (though pandas is primarily used for consistency).

### 2. Text Preprocessing (Steps 4-16) ✨
This is the most extensive section, focusing on cleaning and normalizing the raw text data.

-   **Lowercasing:** Converts all text to lowercase for uniformity.
-   **HTML Tag Removal:** Strips `HTML` tags like `<br />`.
-   **URL Removal:** Eliminates URLs and hyperlinks.
-   **Punctuation Removal:** Removes all punctuation, keeping only alphabetic characters.
-   **Text Normalization:** 
    -   Reduces repeating characters (e.g., `awesooomeee` -> `awesoomee`).
    -   Standardizes whitespace.
-   **Spelling Correction (SymSpellPy):** Uses `SymSpellPy` with a frequency dictionary to correct spelling errors. Error handling is included for robust execution. 
-   **SpaCy Integration:**
    -   **Stopword Removal:** Uses `SpaCy`'s `en_core_web_sm` model to remove common stopwords, preserving negation words crucial for sentiment analysis. Parallelized for speed.
    -   **Tokenization:** Breaks down text into individual words using a blank `SpaCy` model, also parallelized.
    -   **Lemmatization:** Converts words to their base forms (e.g., `running` -> `run`, `worst` -> `bad`) using `SpaCy`'s `en_core_web_sm` model, with optimized component disabling.

### 3. Data Checkpointing (Steps 17-19) 💾

-   **Save Processed Data:** The fully processed DataFrame (with all intermediate cleaning columns) is saved to `movie_reviews_preprocessing_checkpoint.csv`.
-   **Save Lemmatized Reviews:** A clean version, containing only the final `lemmatized_review` column, is saved to `movie_reviews_lematized.csv` for direct use in vectorization.

### 4. Word Frequency Analysis (Steps 20-21) 📈

-   **Word Count:** Calculates the total number of words in the lemmatized corpus.
-   **Vocabulary Analysis:** Uses `collections.Counter` to determine unique words and displays the top 10 most frequent words.

### 5. Numerical Representation (Vectorization) (Steps 22-45) 📊
This section transforms the cleaned text into numerical features.

-   **One-Hot Encoding (OHE) (Steps 22-27):**
    -   Uses `CountVectorizer` with `binary=True` to create a presence/absence matrix.
    -   `max_features` limits the vocabulary size. The sparse matrix is saved and loaded to demonstrate efficient handling.

-   **Bag of Words (BoW) - Unigrams (Steps 28-32):**
    -   Uses `CountVectorizer` (`binary=False`) for word counts.
    -   Analyzes vocabulary, matrix shape, sparsity, and a sample of the matrix.

-   **Bag of Bigrams (Steps 33-36):**
    -   `CountVectorizer` with `ngram_range=(2, 2)` to extract two-word phrases.
    -   Top bigrams are displayed, and the resulting sparse matrix is saved.

-   **Bag of Trigrams (Steps 37-41):**
    -   `CountVectorizer` with `ngram_range=(3, 3)` for three-word phrases.
    -   Top trigrams are displayed, and the sparse matrix is saved.

-   **Master Vectorizer (Step 40):**
    -   A `CountVectorizer` with `ngram_range=(1, 3)` combines unigrams, bigrams, and trigrams into a single feature set (`master_matrix`) for comprehensive modeling.

-   **TF-IDF (Term Frequency-Inverse Document Frequency) (Steps 42-45):**
    -   Uses `TfidfVectorizer` with `ngram_range=(1, 2)` (unigrams and bigrams) and `sublinear_tf=True`.
    -   Inspects vocabulary mapping and analyzes IDF scores to understand word importance.

### 6. Saving All Vectorized Data (Steps 46) 🗜️

-   **Save Sparse Matrices:** All generated sparse matrices (OHE, BoW, Bigrams, Trigrams, TF-IDF) are saved to `.npz` files using `scipy.sparse.save_npz` for efficient storage and retrieval. The TF-IDF vocabulary is also saved to a text file.

---

## 🤝 Contribution

Feel free to fork this repository, open issues, or submit pull requests. Any contributions to improve the preprocessing, add new vectorization methods, or integrate machine learning models are welcome!

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE-optional-if-you-have-one.md).
