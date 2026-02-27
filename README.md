# GAVELbench

Benchmark for å evaluere hvor godt LLM-er svarer på spørsmål om Navs tjenester, ved hjelp av ekte spørsmål og referansesvar fra Bob (Navs KI-støtte til kontaktsenteret).

MERK: Denne filen er autogenerert fra kodeagent.

## Hva programmet gjør

1. **Genererer svar** — sender spørsmål fra Bob til én eller flere LLM-er og lagrer svarene i `data/generated/`.
2. **Evaluerer svar** — sammenligner genererte svar med Bobs referansesvar ved hjelp av flere metrikker.
3. **Rapporterer resultater** — skriver en Markdown-rapport med sammenligningstabell, radarplot og kommentarer til `data/results/`.

## Oppsett

Krever Python ≥ 3.12 og [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Google Cloud-autentisering er nødvendig for tilgang til Vertex AI-modeller:

```bash
gcloud auth login --update-adc
```

## Kjøre analysen

### Full pipeline (henting av data + generering + evaluering)

```bash
uv run python src/main.py
```

Pipelinen kjøres i tre steg:

| Steg | Beskrivelse |
|------|-------------|
| 0 | Henter Bob-spørsmål og -svar fra BigQuery → `data/bob_data.jsonl` |
| 1 | Genererer svar fra alle modeller i `models.yaml` → `data/generated/` |
| 2 | Evaluerer svarene og skriver rapport → `data/results/` |

### Hopp over steg

```bash
# Hopp over BQ-henting (bruk eksisterende bob_data.jsonl)
uv run python src/main.py --skip-fetch

# Hopp over både BQ-henting og generering (kun evaluering)
uv run python src/main.py --skip-generation
```

### Kjøre enkeltsteget

```bash
# Kun henting av Bob-data fra BigQuery
uv run python src/load_data_from_BQ.py

# Kun generering av svar
uv run python src/gen_answers_from_llm.py

# Kun evaluering
uv run python src/evaluate.py
```

Resultater skrives til `data/results/`.


### Metrikker

| Metrikk | Beskrivelse | Retning |
|---|---|---|
| ROUGE-L | Lengste felles delsekvens F1 mellom generert og referansesvar | ↑ høyere er bedre |
| BERTScore F1 | Kontekstuell embedding-likhet (`bert-base-multilingual-cased`) | ↑ høyere er bedre |
| Cosinuslikhet | TF-IDF vektorcosinuslikhet | ↑ høyere er bedre |
| Jensen-Shannon-divergens | Avstand mellom sannsynlighetsfordelinger over ordforråd | ↓ lavere er bedre |
| NLI entailment | Sannsynlighet for at det genererte svaret impliserer referansesvaret (`alexandrainst/scandi-nli-small`) | ↑ høyere er bedre |

Nye metrikker legges til ved å dekorere en funksjon med `@register_metric("navn")` i `src/evaluate.py` — ingen andre kodeendringer nødvendig.

## Data

```
data/
├── bob_data.jsonl          ← fasitsvar og spørsmål fra Bob
├── generated/              ← genererte svar, én fil per modell
│   └── generated_answers_<modell-id>.jsonl
└── results/                ← evalueringsresultater
    ├── evaluation_report.md
    ├── evaluation_report.json
    └── evaluation_report_radar.png
```

Hver `generated_answers_*.jsonl`-fil har rader med feltene `question` og `answer`.
`bob_data.jsonl` har feltene `contextualized_question` og `answer_content`.

## Modeller

Modellene konfigureres i `src/models.yaml`. Å legge til en ny modell krever kun å endre yaml-en, ingen kodeendringer:

```yaml
defaults:
  project: <gcp-prosjekt>
  location: europe-west1

models:
  - id: gemini-2.0-flash-001
    provider: vertex_ai
    description: Rask og kostnadseffektiv Gemini-modell
```

Hvert innslag arver `project` og `location` fra `defaults` med mindre det overstyres. Støttet tilbyder foreløpig: `vertex_ai`.
For å legge til en ny tilbyder, legg til en branch i `generate_answer()` i `src/gen_answers_from_llm.py`.

## Utvikling

```bash
# Lint og typesjekk
just lint

# Automatisk fiks av formatering
just fix
```
