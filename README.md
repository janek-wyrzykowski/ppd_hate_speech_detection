# Detekcja mowy nienawiści

Projekt polegający na detekcji mowy nienawiści w komentarzach z Wikipedii. Każdy komentarz jest klasyfikowany pod kątem:
- toksyczności,
- drastycznie toksycznych,
- nieprzyzwoitości,
- obecności gróźb,
- obecności obelg,
- nienawiści ze względu na tożsamość.

Celem tego projektu jest stworzenie kilku modeli klasyfikujących komentarze w kategorii toksyczności oraz porównanie skuteczności modelu ze skutecznością popularnych LLMów.

### Wymagania

W aktualnej wersji wymagane jest ręczne pobranie danych ze [strony internetowej Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data). Wypakowane pliki `.csv` należy umieścić w folderze `data`.