# GAVELbench

Benchmark for å evaluere hvor godt LLM-er svarer på spørsmål om Navs tjenester, ved hjelp av ekte spørsmål og referansesvar fra Bob (Navs KI-støtte til kontaktsenteret).

## Hva programmet gjør

Pipelinen er delt i fire tydelige steg:

| Steg | Modul | Beskrivelse |
|------|-------|-------------|
| 0 | `src/fetch_data.py` | Henter Bob-spørsmål og -svar fra BigQuery, sampler per overkategori |
| 1 | `src/generate.py` | Genererer svar fra alle modeller i `models.yaml` |
| 2 | `src/evaluate.py` | Evaluerer svarene mot referansesvar med flere metrikker |
| 3 | `src/report.py` | Skriver JSON-resultater, radar- og søyleplott, og statiske Quarto-sider |

## Oppsett

Krever Python ≥ 3.12 og [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For lokale HuggingFace-modeller (GGUF) på Apple Silicon (M1/M2/M3), installer med Metal-støtte:

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv sync --group huggingface
```

Google Cloud-autentisering er nødvendig for tilgang til Vertex AI-modeller:

```bash
gcloud auth login --update-adc
```

## Kjøre analysen

### Full pipeline

```bash
uv run python src/main.py
```

### Hopp over steg

```bash
# Hopp over BQ-henting (bruk eksisterende kunnskapsbase_kategorier.json)
uv run python src/main.py --skip-bq-fetch

# Hopp over BQ-henting og generering (kun evaluering + rapport)
uv run python src/main.py --skip-generation

# Regenerer kun rapport og plott fra eksisterende JSON-resultater
uv run python src/main.py --only-report
```

### Kjøre enkeltsteget

```bash
# Hente Bob-data fra BigQuery
uv run python src/fetch_data.py

# Generere svar fra alle modeller
uv run python src/generate.py

# Evaluere og skrive rapport
uv run python src/evaluate.py
```

### Quarto-rapport

Etter at pipelinen har kjørt kan du forhåndsvise eller bygge Quarto-rapporten:

```bash
# Forhåndsvisning
just preview

# Bygg til HTML
just render
```

QMD-filene er statiske (ingen Python-kode) og genereres automatisk av `src/report.py`. De trenger ikke redigeres manuelt.

### Metrikker

| Metrikk | Beskrivelse | Retning |
|---|---|---|
| ROUGE-L | Lengste felles delsekvens F1 | ↑ høyere er bedre |
| BERTScore F1 | Kontekstuell embedding-likhet (`bert-base-multilingual-cased`) | ↑ høyere er bedre |
| Cosinuslikhet | TF-IDF vektorcosinuslikhet | ↑ høyere er bedre |
| Jensen-Shannon-divergens | Avstand mellom ordfordelinger | ↓ lavere er bedre |
| NLI entailment | Sannsynlighet for at generert svar impliserer referansesvaret (`alexandrainst/scandi-nli-base`) | ↑ høyere er bedre |

Nye metrikker legges til ved å dekorere en funksjon med `@register_metric("navn")` i `src/evaluate.py`.

## Kodestruktur

```
src/
├── models.yaml          # Modellkonfigurasjon
├── main.py              # CLI-inngangspunkt for hele pipelinen
├── models.py            # Delt: ModelConfig, load_model_configs(), active_model_ids()
├── fetch_data.py        # Steg 0: hente og sample data fra BigQuery
├── generate.py          # Steg 1: generere svar fra LLM-er
├── evaluate.py          # Steg 2: beregne metrikker (returnerer data, skriver ingenting)
└── report.py            # Steg 3: skrive JSON, plott og statiske QMD-filer

tests/
└── test_api_access.py   # Røyktest for API-tilgang til alle modeller

scripts/
├── explore_data_categories.py  # Utforskning av kategoridistribusjon
└── ragas_evaluate.py           # RAGAS-basert faktakorrekthetsevaluering

quarto/
├── index.qmd            # Autogenerert (statisk) — kjør --only-report for å oppdatere
└── per_kategori.qmd     # Autogenerert (statisk) — kjør --only-report for å oppdatere
```

## Data

```
data/
├── kunnskapsbase_kategorier.json  ← rådata fra BigQuery (privat, ikke i git)
├── kategorier_mapping.json        ← mapping kategori → overkategori (i git)
├── bob_data.json                  ← samplet treningssett (privat, ikke i git)
├── generated/                     ← genererte svar per modell (privat, ikke i git)
└── results/                       ← evalueringsresultater
    ├── evaluation_report.json              ← aggregerte scores (i git)
    ├── evaluation_report_scores_per_overkategori.json  ← per-kategori scores (i git)
    └── *.png                               ← radar- og søyleplott (i git)
```

## Modeller

Modellene konfigureres i `src/models.yaml`. Å legge til en ny modell krever kun å endre yaml-en:

```yaml
defaults:
  project: <gcp-prosjekt>
  location: europe-west1

models:
  - id: gemini-2.0-flash
    provider: vertex_ai
    description: Rask og kostnadseffektiv Gemini-modell
    concurrency: 10
```

Støttede tilbydere: `vertex_ai`, `vertex_anthropic`, `huggingface`.
For å legge til en ny tilbyder, legg til en branch i `run_model()` i `src/generate.py`.

## Utvikling

```bash
# Lint og typesjekk
just lint

# Automatisk fiks av formatering
just fix

# Test API-tilgang for alle modeller i models.yaml
uv run python tests/test_api_access.py
```
