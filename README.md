# Detekcja mowy nienawiści

Projekt polegający na detekcji mowy nienawiści w komentarzach z Wikipedii. Każdy komentarz jest klasyfikowany pod kątem:
- toksyczności,
- drastycznie toksycznych,
- nieprzyzwoitości,
- obecności gróźb,
- obecności obelg,
- nienawiści ze względu na tożsamość.

Celem tego projektu jest stworzenie kilku modeli klasyfikujących komentarze w kategorii toksyczności oraz porównanie skuteczności modelu ze skutecznością popularnych LLMów.

### Środowisko

W celu utrzymania odtwarzalnego środowiska w tym projekcie wykorzystany jest `uv`. Należy go zainstalować za pomocą jednej z poniższych komend:

```sh
# Linux/MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# lub
wget -qO- https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatywne metody:

```sh
pip install uv
```

```sh
brew install uv
```

```powershell
# Windows
winget install --id=astral-sh.uv  -e
```

Następnie należy zsynchronizować środowisko za pomocą komendy

```
uv sync
```

W celu dodania nowych pakietów należy wykonać poniższe komendy:

```
uv add <nazwa-pakietu>
uv lock
```

### Dane

Wykorzystane w projekcie dane zostały pobrane ze strony https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data i znajdują się one w folderze `data/raw`. Modele są jednak trenowane, walidowane i testowane na wybranych podzbiorach tych danych. Zostały one przygotowane z pomocą skryptu `scripts/split.ipynb` i znajdują się w folderze `data/selected`.