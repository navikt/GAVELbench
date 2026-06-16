# Kilderegister: benchmarker og evalueringskilder for ansvarlig KI

Vedlegg til [veiledningen om benchmarker ved modellvalg](ansvarlig-ki-benchmarker.md).

**Sist gjennomgått:** juni 2026
**Revideres:** halvårlig, eller når en kilde endrer metodikk, navn eller dekning

Registeret er øyeblikksbilde, ikke fasit. Sjekk alltid dato og modellversjon i kilden selv før dere bruker et resultat.

## Kilder per område

| Område | Kilde | Hva den dekker                                                                                                                     | Modelltyper | Evidenstype | Status juni 2026                                                      |
|---|---|------------------------------------------------------------------------------------------------------------------------------------|---|---|-----------------------------------------------------------------------|
| Helhetlig vurdering | [HELM](https://crfm.stanford.edu/helm/) (Stanford CRFM) | Flere scenarioer, metrikker og modeller i ett rammeverk; egne ledertavler for kapasitet, sikkerhet og multimodalitet               | Proprietære og åpne der de er inkludert | Uavhengig rammeverk | Levende; sjekk dato og versjon                                        |
| Sikkerhet og skadelig innhold | [HELM Safety](https://crfm.stanford.edu/helm/safety/latest/) | Diskriminering, vold, seksuelt innhold, svindel, trakassering, desinformasjon, skadelig rådgivning                                 | Proprietære og åpne der de er inkludert | Uavhengig benchmark | Levende ledertavle                                                    |
| Regelverksnær sikkerhet | [AIR-Bench](https://crfm.stanford.edu/helm/air-bench/latest/) | Sikkerhetskategorier avledet av policy- og regelverkskategorier                                                                    | Proprietære og åpne der de er inkludert | Uavhengig benchmark | Levende ledertavle                                                    |
| Skadelig innhold | [AILuminate](https://mlcommons.org/benchmarks/ailuminate/) (MLCommons) | Chat-sikkerhet på tvers av 12 farekategorier, flere språk, jailbreak-varianter                                                     | Modeller/systemer som er sendt inn eller testet i oppsettet | Innsendt/tredjeparts | Dekningen utvikles; sjekk gjeldende versjon                           |
| Leverandøråpenhet | [Foundation Model Transparency Index](https://hai.stanford.edu/news/transparency-in-ai-is-on-the-decline) (Stanford HAI) | Leverandørers åpenhet om treningsdata, risiko, miljø og bruk                                                                       | Utvalgte modellutviklere, ikke enkeltmodeller | Leverandørindeks | 2025-utgaven viser fallende gjennomsnittlig åpenhet                   |
| Energi | [AI Energy Score](https://huggingface.co/blog/sasha/ai-energy-score-v2) | Energieffektivitet per oppgave og modellklasse, 1–5 stjerner; v2 dekker resonneringsmodeller                                       | I hovedsak åpne/nedlastbare modeller; proprietære bare der de sendes inn | Standardisert energibenchmark | v2 lansert desember 2025                                              |
| Norsk språk | [NorEval](https://github.com/ltgoslo/noreval) (LTG, UiO) | Norsk språkforståelse og generering på bokmål og nynorsk: kunnskap, leseforståelse, sammendrag, instruksjonsfølging, sannferdighet | Primært åpne modeller i reproduserbart oppsett | Norsk språkbenchmark | Fra 2025; mest dekkende norske benchmark                              |
| Europeiske språk | [EuroEval](https://euroeval.com/) (tidligere ScandEval) | Språkmodeller på 30+ europeiske språk, inkludert norsk; offentlig leaderboard og reproduserbart oppsett                            | Flere modelltyper, avhengig av leaderboard | Språkbenchmark | Levende; ScandEval-navnet er utgått (samme prosjekt, utvidet dekning) |
| Enkeltegenskaper | Komponentbenchmarker som BBQ og TruthfulQA, ofte via HELM | Skjevhet, sannferdighet o.l. som enkeltegenskaper, i hovedsak på engelsk, med amerikanske demografiske kategorier               | Varierer | Svakt til middels grunnlag alene | Endrer seg raskt; verifiser før bruk                                  |


## Leverandørrapporterte energitall

Tallene under er leverandørspesifikke, selvrapporterte og målt med ulik metodikk og systemgrense. De kan ikke sammenlignes direkte på tvers.

| Leverandør | Hva som finnes | Forbehold                                                                                                                                                                                 |
|---|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Google | [Rapporterer](https://cloud.google.com/blog/products/infrastructure/measuring-the-environmental-impact-of-ai-inference/) at en median tekstprompt i Gemini Apps bruker ca. 0,24 Wh, 0,03 gCO₂e og 0,26 ml vann (aug. 2025) | Gjelder median tekstprompt i Gemini Apps, ikke nødvendigvis Vertex, lange kontekster, multimodale kall, resonnering eller Navs bruk. Median, ikke gjennomsnitt; trening ikke inkludert |
| Anthropic | Begrenset offentlig per-prompt-informasjon | Ikke sammenlignbart uten felles metodikk                                                                                                                                                  |
| OpenAI | Enkelte grove offentlige anslag | Ulik metodikk og systemgrense; lite fullstendig dokumentasjon                                                                                                                             |
| Åpne modeller | AI Energy Score gir standardisert sammenligning for modeller som inngår | Dekker ikke alle modeller, oppsett eller lokal driftsoptimalisering                                                                                                                       |

## Hull i dekningen (kjente, per juni 2026)

Disse hullene er funn i seg selv.  Dokumentér dem som usikkerhet i vurderinger:

- **Norske diskrimineringsakser.** Ingen kjent benchmark dekker samiske brukere, nasjonale minoriteter, norsk som andrespråk eller norske klassemarkører. Bias-benchmarker som BBQ er bygget på amerikanske kategorier.
- **Tilgjengelighetsrelevante modellegenskaper.** Klarspråkproduksjon og robusthet mot ikke-standard input er i praksis ikke benchmarket.
- **Energi for proprietære API-modeller.** Standardiserte mål finnes i hovedsak for åpne modeller; for proprietære er dere henvist til leverandørenes egne tall.
- **Norsk forvaltningsspråk og -kontekst.** Språkbenchmarkene måler generell norsk, ikke forvaltningsspråk, regelverkstekst eller Nav-relevante oppgavetyper.

Finner dere en kilde som tetter et av hullene, eller oppdager at en oppføring er utdatert, si fra på [#tada](https://nav-it.slack.com/archives/C03CXENSLMV)!
