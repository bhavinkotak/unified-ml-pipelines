# Data sources

All files are cached by `code/run_all.py` via `sklearn.datasets.fetch_openml` and verified by sha256.

| File | Dataset | URL | License | Retrieved (UTC) | sha256 |
|---|---|---|---|---|---|
| data/raw/adult.csv.gz | adult (OpenML 1590) | https://www.openml.org/d/1590 | CC BY 4.0 (UCI Machine Learning Repository); OpenML licence field: Public | 2026-11-05 | `ac42c3f748a6b621dabab83d2089f606695ef91bf81bd15a9e7df347d6cced09` |
| data/raw/bank-marketing.csv.gz | bank-marketing (OpenML 1461) | https://www.openml.org/d/1461 | CC BY 4.0 (UCI Machine Learning Repository); OpenML licence field: Public | 2026-11-05 | `fa7691e699c46c8cd6111b83e097084247e6e472e3b4cabbc019c2d40f4a5294` |
| data/raw/credit-g.csv.gz | credit-g (OpenML 31) | https://www.openml.org/d/31 | CC BY 4.0 (UCI Machine Learning Repository); OpenML licence field: Public | 2026-11-05 | `4ff6ec7c5f1fa979c45f19570df0f6cbe783d2b2bd9d424a40f31f1a8ed48c5d` |

Original sources: UCI Adult (https://archive.ics.uci.edu/dataset/2/adult), UCI Bank Marketing (https://archive.ics.uci.edu/dataset/222/bank+marketing), UCI Statlog German Credit (https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
Bank Marketing column V12 (call duration) is excluded because it is only known after the outcome.
