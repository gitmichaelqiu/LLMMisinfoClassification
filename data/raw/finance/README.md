# Finance Raw Data

Source files derived from the [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php).

| File | Contents | Rows | Tracked |
|---|---|---|---|
| `finance_test_500.csv` | Balanced test set (250 REAL, 250 FAKE) | 500 | Yes |
| `finance_corpus.csv` | TF-IDF retrieval pool | 2,756 | Yes |
| `financial_news.csv` | Original economics-filtered articles | 3,324 | No |
| `financial_news_clean.csv` | Deduplicated variant of original | 3,731 | No |

Columns: `title`, `text`, `label` (0=REAL, 1=FAKE).

Loaded by `src/data.py::load_all_data()`.
