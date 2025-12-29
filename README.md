# Detekcja Mowy Nienawiści - Projekt PPD

Projekt polegający na detekcji mowy nienawiści w komentarzach z Wikipedii. Komentarze są klasyfikowane w 3 kategoriach:

- **0**: nietoksyczny (non-toxic)
- **1**: toksyczny (toxic)
- **2**: bardzo toksyczny (severely-toxic)

**Modele porównywane:**

- Baseline: TF-IDF + Logistic Regression
- XGBoost: Embeddings + XGBoost
- BERT: Fine-tuned BERT
- LLM: Groq API (Llama 3.1 70B)

**Metryki:** Accuracy, Macro-F1 (główna), ROC-AUC, Confusion Matrix

## Uruchomienie

Główny notebook projektu: [`notebooks/ppd_projekt.ipynb`](notebooks/ppd_projekt.ipynb)

Zawiera kompletny pipeline:

1. EDA - analiza danych
2. Przygotowanie danych i split
3. Trenowanie wszystkich modeli
4. Ewaluacja i porównanie
5. Analiza błędów

### Wymagania

```bash
# Zainstaluj uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Zsynchronizuj środowisko
uv sync

# Dla LLM (opcjonalne) - ustaw klucz API
export GROQ_API_KEY="your-key-here"
```

### Struktura projektu

```
├── data/                    # Dane treningowe/testowe
│   ├── raw/                # Surowe dane z Kaggle
│   ├── df_train.csv        # Zbiór treningowy
│   ├── df_val.csv          # Zbiór walidacyjny
│   ├── df_test.csv         # Zbiór testowy
│   └── df_test_sample.csv  # Próbka testowa (500 próbek)
├── notebooks/
│   ├── ppd_projekt.ipynb   # Główny notebook
│   ├── BERTpred*.txt       # Predykcje BERT
│   └── BERTA*.py          # Skrypty BERT
└── reports/               # Wyniki i wykresy
```

### Wyniki

Wyniki porównania modeli są dostępne w folderze `reports/` po uruchomieniu notebooka.
