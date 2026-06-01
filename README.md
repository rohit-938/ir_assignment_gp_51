# 📚 Information Retrieval — Assignment 1

> **Course:** AIMLCZG537 / DSECLZG537 &nbsp;|&nbsp; **Semester:** S2-2025 &nbsp;|&nbsp; **Marks:** 10 &nbsp;|&nbsp; **Deadline:** 15 June 2025

An end-to-end Information Retrieval system built with **Streamlit**, covering text preprocessing, indexing, phrase queries, dictionary search structures, and tolerant retrieval — all interactive through the browser.

---

## 📁 Project Structure

```
ir_assignment1/
├── app.py                  # Main Streamlit application (entry point)
├── preprocessing.py        # Tokenization, stopwords, stemming, lemmatization
├── indexing.py             # Inverted, biword, and positional index builders
├── phrase_query.py         # Biword and positional query processors
├── tree_structures.py      # BST and B-Tree with timing benchmarks
├── tolerant.py             # Wildcard, spelling, k-gram, phonetic retrieval
├── utils.py                # Shared helpers — metrics, timing, similarity
├── data/                   # Document collection (.txt or .csv files)
├── setup.py                # One-command setup and launcher
├── requirements.txt        # All pip dependencies with pinned versions
└── README.md               # This file
```

---

## ⚡ Quick Start (any machine)

> **Requires Python 3.10 or higher.** Download from [python.org](https://www.python.org/downloads/) if not installed.

### 1 — Clone or copy the project

```cmd
cd D:\your\folder
```

Make sure the project folder contains `setup.py`, `requirements.txt`, and `app.py`.

### 2 — Run setup (one command does everything)

```cmd
python setup.py
```

This will automatically:
- ✅ Check your Python version
- ✅ Create a virtual environment (`ir_env/`)
- ✅ Upgrade pip
- ✅ Install all libraries from `requirements.txt`
- ✅ Download the spaCy English model
- ✅ Download all NLTK data packages
- ✅ Launch the Streamlit app in your browser

> The browser opens at **http://localhost:8501** automatically.

---

## 🔁 Running Again Later

After the first setup, you don't need to run `setup.py` again. Just activate the venv and start the app:

**Windows:**
```cmd
ir_env\Scripts\activate
streamlit run app.py
```

**Mac / Linux:**
```bash
source ir_env/bin/activate
streamlit run app.py
```

Or just run `python setup.py` again — it skips steps already done.

---

## 📦 Dependencies

All libraries are pinned in `requirements.txt`. Key ones:

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.45.0 | Front-end UI framework |
| `nltk` | 3.9.1 | Tokenization, stemming, lemmatization |
| `spacy` | 3.8.4 | Advanced lemmatization |
| `scikit-learn` | 1.6.1 | TF-IDF, cosine similarity |
| `pandas` | 2.2.3 | Tabular results display |
| `numpy` | 1.26.4 | Numerical operations |
| `rapidfuzz` | 3.10.1 | Edit distance, spelling correction |
| `jellyfish` | 1.1.0 | Phonetic matching (Soundex, Metaphone) |
| `sortedcontainers` | 2.4.0 | B-Tree approximation |

To regenerate `requirements.txt` from your environment:
```cmd
pip freeze > requirements.txt
```

---

## 🖥️ App Features

The Streamlit app is organised into **6 tabs**:

| Tab | Feature | Assignment Task |
|---|---|---|
| 📂 Upload & Preview | Upload documents, preview collection | Task A |
| 🔤 Preprocessing | Tokenize, lowercase, remove stopwords, stem, lemmatize | Task B |
| 🔍 Phrase Query | Biword index vs positional index comparison | Task C |
| 🌳 Dictionary Search | BST vs B-Tree timing benchmarks | Task D |
| 🧩 Tolerant Retrieval | Wildcard, spelling correction, k-gram, phonetic | Task E |
| 📝 Inference | Written conclusions and experimental results | Task G |

---

## 📊 Assignment Components

| Component | Marks |
|---|---|
| Streamlit end-to-end workflow | 1.0 |
| Text preprocessing | 1.5 |
| Stemming vs lemmatization comparison | 1.0 |
| Phrase query (biword + positional) | 1.5 |
| BST vs B-Tree dictionary search | 1.5 |
| Tolerant retrieval | 1.5 |
| Experimental evidence & inference | 1.0 |
| Virtual Lab usage | 1.0 |
| **Total** | **10** |

---

## 🗂️ Dataset

Place your document collection inside the `data/` folder as `.txt` or `.csv` files.

**Recommended datasets:**
- [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) — load via `sklearn.datasets.fetch_20newsgroups()`
- [Reuters-21578](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection) — classic IR benchmark
- Your own `.txt` files on any topic (50–200 documents recommended)

---

## 🛠️ Manual Setup (if setup.py fails)

If you prefer to set up manually:

```cmd
python -m venv ir_env
ir_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
streamlit run app.py
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `python setup.py` — Python not found | Install Python 3.11 from python.org and tick **Add to PATH** |
| `No module named streamlit` | Make sure venv is activated before running |
| spaCy model not found | Run `python -m spacy download en_core_web_sm` inside venv |
| NLTK data missing at runtime | Run the `nltk.download(...)` command in manual setup above |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| `'python3.11' is not recognized` | Use `py -3.11` on Windows instead |

---

## 📌 Notes

- All experiments and inferences must be visible inside the Streamlit UI.
- Run the app inside the **BITS Virtual Lab** and take screenshots for the 1-mark virtual lab component.
- Post questions on the **Taxila Discussion Forum** — check existing threads before posting.

---

*BITS Pilani — Information Retrieval (AIMLCZG537 / DSECLZG537) — S2 2025*
