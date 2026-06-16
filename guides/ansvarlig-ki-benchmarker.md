# Ansvarlig KI: hvordan bruke benchmarker når dere velger modell

Veiledning til team som skal velge, sammenligne eller dokumentere bruk av språkmodeller.  Fremfor å peke ut én «trygg» modell, er hensikten med teksten å hjelpe dere å bruke benchmarkresultater til det de duger til: screening og beslutningsstøtte.

## Kortversjonen

En benchmark er bare relevant når den treffer det dere faktisk skal bygge: samme modell og versjon, samme språk, samme type oppgave, omtrent samme bruksmønster. En høy skår på en generell engelsk chat-benchmark sier lite om en norsk RAG-løsning for saksbehandling. Den sier bare at modellen gjorde det bra på akkurat den testen.

Ansvarligheten ligger i brukstilfellet, ikke i modellen. Hvilke data løsningen bruker, hvem den påvirker, hvilken beslutning den inngår i, og hvilken menneskelig kontroll som finnes--det er dét som avgjør om bruken er forsvarlig. En modell som skårer godt på en rettferdighetsbenchmark kan fortsatt inngå i et diskriminerende system. Skjevheten oppstår like gjerne i dataene, tersklene eller saksflyten som i selve modellen.

Det vil si at en forsvarlig konklusjon ser slik ut:

> Vi har valgt modell X versjon Y fordi den løser oppgaven godt nok i vår lokale evaluering, har akseptabel norsk ytelse og dokumentert modellinformasjon, og ikke viser uakseptabel risiko i våre tester. Disse eksterne benchmarkene er brukt som støtte, ikke som attest. Disse hullene gjenstår, og håndteres med tiltak A, B og C.

Ikke slik:

> Modell X er trygg fordi den ligger høyt på benchmark Y.



## Det dette dokumentet dekker

Veiledningen dekker kun ansvarlig-KI-egenskapene ved modellvalg. Kostnad, kapasitet, latens, drift og arkitektur må dere vurdere i tillegg.

Modellene dere velger blant, kjører i hovedsak på Gemini Enterprise Agent Platform (tidligere Vertex AI). Det betyr at noen spørsmål allerede er avklart på plattformnivå, én gang for alle team:

- databehandleravtale og hvor dataene behandles
- tilgangsstyring, logging og sikkerhetsoppsett på plattformnivå
- kontraktsvilkår mot leverandøren

Dere trenger ikke gjøre denne vurderingen på nytt per modell. Det dere *eier*, er alt som avhenger av deres kontekst: valget av modell innenfor plattformen, om modellen passer oppgaven, den lokale evalueringen, og dokumentasjonen av beslutningen.

Er dere usikre på hvor grensen går i et konkret tilfelle, spør **XXXXX** før dere antar at noe er dekket.

Én konsekvens av plattformvalget er verdt å merke seg: «åpen modell» betyr i vår praksis vanligvis *åpne vekter på en forvaltet plattform*, ikke selvhosting i egen infrastruktur. Det gir dere fordelen ved åpne modeller (for eksempel at hvem som helst kan teste dem, så de har oftere rikere og mer uavhengig benchmarkdekning enn proprietære modeller) uten at dere arver driftsansvaret. Men en åpen modell er mer *testbar*, ikke automatisk *tryggere*. Forskjellen er viktig.

## Prosessen

Fem steg, i denne rekkefølgen. Stegene bygger på hverandre: hopper dere over det første, blir resten gjetting.

### 1. Beskriv brukstilfellet presist.

Skriv kort ned hvem brukeren er, hva modellen skal og ikke skal gjøre, hvilke data den får tilgang til, hva en alvorlig feil er, hva en akseptabel feil er, og hvem som oppdager og retter feil. Er ikke dette klart, er det for tidlig å se på benchmarkskårer. Dere vet ennå ikke hva en skår skulle vært relevant *for*.

### 2. Velg hvilke områder dere må vurdere.

Tre områder gjelder for praktisk talt alle Nav-relevante brukstilfeller. Dem velger dere ikke bort:

- **Norsk språk.** Denne er en forutsetning for at alle andre vurderinger gir mening. En modell som misforstår norske henvendelser kan ikke vurderes for skjevhet eller sannferdighet på norsk.
- **Skjevhet og diskriminering.** Den mest rettighetskritiske aksen i offentlig sektor, og den dårligst dekkede i generelle benchmarker.
- **Personvern og lekkasje.** Risiko for gjengivelse av treningsdata, personopplysninger eller informasjon fra kontekst som ikke skulle brukes.

Resten utløses av brukstilfellet, ikke bare hva som teamet synes er interessant.

Bruk disse utløserne:

| Hvis løsningen ... | ... må dere også vurdere |
|---|---|
| møter innbyggere direkte | tilgjengelighet, sårbare brukere, eskalering |
| bruker dokumenter eller kilder (RAG) | kildeforankring, sitering, foreldede kilder |
| gir veiledning brukeren kan handle på | sannferdighet, kalibrering, medgjørlighet |
| klassifiserer, prioriterer eller støtter beslutninger | feilrater per gruppe, terskler, overstyring |
| kan møte samisk eller andre språk enn norsk | språkvarianter og hva som skjer når modellen ikke forstår |
| skal kjøre i stort volum eller bruke resonneringsmodeller | energi og ressursbruk |

Grunnen til at utløsere er bedre enn interesse: team tester gjerne hallusinasjon, fordi det er morsomt, og hopper stille over diskriminering, fordi det er ubehagelig.  Hvert område er beskrevet nærmere lenger ned.

### 3. Finn eksterne signaler og vurder om de gjelder dere.

Relevante benchmarker og kilder per område står i [kilderegisteret](./kilderegister-benchmarker.md). Kort om kildetypene:

- **Uavhengige rammeverk** som HELM gir sammenlignbare resultater, men bare for modellene og versjonene som faktisk er testet.
- **Innsendingsbaserte suiter** som AILuminate er nyttige når riktig system er testet; dekningen avhenger av hva som er sendt inn.
- **Leverandørens modellkort og systemkort** er nyttig dokumentasjon, men behandle dem som leverandørpåstander, ikke uavhengig måling.
- **Språkbenchmarker** som NorEval og EuroEval er viktig for oss, men måler språkytelse, ikke ansvarlig-KI-risiko direkte.

For hver kilde dere bruker: noter modellversjon, dato, språk, og om kilden er uavhengig, innsendt eller selvrapportert.

Vurder så overføringsverdien. Spør:

- Er modellen og versjonen den samme som dere skal bruke?
- Er språket norsk?
- Måler testen grunnmodellen, eller systemet (chat, RAG, agent, guardrails)?
- Ligner oppgaven på deres?

Er svaret nei på flere av disse, behandle benchmarken som et svakt signal. En god skår på feil modell, feil versjon, feil språk eller feil oppgave gir falsk trygghet.  Den *føles* som dokumentasjon uten å være det.

### 4. Gjør lokal evaluering.

Eksterne benchmarker erstatter ikke lokal evaluering. De forteller dere hva dere bør teste selv, ikke hva resultatet blir.

Lag et lite, representativt norsk evalueringssett fra egen oppgavekontekst, uten personopplysninger og uten interne data som ikke kan brukes til testformål. For de fleste språkmodelltilfeller bør dere minst teste:

- norsk språk og oppgaveforståelse, inkludert nynorsk der det er relevant
- hallusinasjon og overkonfidens
- lekkasje fra kontekst
- prompt injection og manipulerende input
- overavvisning og underavvisning
- om feil rammer ulike brukergrupper ulikt
- input som ikke er standard bokmål (skrivefeil, norsk som andrespråk, dialektnær skrift, tekst fra talegjenkjenning)

For RAG-løsninger i tillegg: om svaret faktisk bygger på riktige kilder, om modellen siterer riktig, om den utelater viktige forbehold, om den blander gammel og ny informasjon, og om den sier fra når kildene ikke gir grunnlag for svar.

For løsninger som påvirker rettigheter, plikter, prioritering eller tilgang til tjenester må evalueringen være strengere og knyttes til risikovurdering og juridisk vurdering. Da er ikke denne veiledningen nok.  Ta gjerne kontakt for videre drøfting.

### 5. Dokumentér beslutningen.

Dokumentér både hvilken modell dere valgte og hvorfor.

Skriv ned modellnavn og versjon, hvilke eksterne kilder dere brukte og når, hva dere testet selv og hva dere fant, hvilke hull som gjenstår med hvilke tiltak, hvem som eier oppfølgingen, og når vurderingen skal opp igjen (for eksempel ved ny modellversjon eller vesentlig endret bruk).

Fant dere ikke norsk benchmark, bias-evaluering, miljødata eller modellkort? Det er også et funn. Skriv det ned som usikkerhet, ikke fyll hullet med antakelser.

## Områdene, og hvorfor de ser ut som de gjør hos oss

### Skjevhet og diskriminering

Dette er lett å misforstå: et team ser «sikkerhetsbenchmark» og tror diskriminering er dekket. Det er ikke nødvendigvis sant. Noen benchmarkoppsett inkluderer bias; andre avgrenser det bort, fordi diskrimineringsskader ofte krever lengre kontekst, konkrete beslutningsregler og kunnskap om hvem som påvirkes.

Hva dere skal teste for, er ikke en smakssak. [Likestillings- og diskrimineringsloven §6](https://lovdata.no/lov/2017-06-16-51/§6) lister diskrimineringsgrunnlagene: kjønn, graviditet og omsorgsoppgaver, etnisitet, religion og livssyn, funksjonsnedsettelse, seksuell orientering, kjønnsidentitet og kjønnsuttrykk, alder, og kombinasjoner av disse. Bruk lista som sjekkliste. Da blir «vi har vurdert diskriminering» en testbar påstand i stedet for en følelse.

Og her er begrensningen i de eksterne kildene: de store bias-benchmarkene, som BBQ, er bygget på *amerikanske* demografiske kategorier, dvs. amerikanske etnisitetsinndelinger, amerikansk religionskontekst, amerikanske stereotypier. Ingen benchmark tester hvordan en modell håndterer samiske brukere, nasjonale minoriteter, norsk som andrespråk, eller norske klassemarkører i språket. En god BBQ-skår sier altså noe om amerikanske skjevheter; den sier omtrent ingenting om de norske. Dette er fravær-av-data-funnet i sin reneste form: dokumentér det, og test de norske aksene selv der brukstilfellet tilsier det.

For borgerrettede eller rettighetsnære løsninger holder det aldri med benchmarker her. Da må dere teste feilrater, terskler og konsekvenser per gruppe i egen kontekst. Spørsmålet er ikke om modellen unngår å produsere eksplisitt hat, det er om systemet gir ulike feil, ulik nytte eller ulik belastning for ulike grupper.

### Norsk språk og språkvarianter

En modell som fungerer godt på engelsk kan fortsatt misforstå norske henvendelser, prestere dårligere på nynorsk enn bokmål, håndtere forvaltningsspråk dårlig, og miste nyanser i rettigheter, vilkår og unntak. NorEval og EuroEval gir nyttige signaler, men de erstatter ikke testing på deres egne oppgaver.

Tenk også på hvilke språk løsningen faktisk vil *møte*, ikke bare hvilke den er ment for. Samiske språk har en særstilling: [sameloven kapittel 3](https://lovdata.no/dokument/NL/lov/1987-06-12-56/KAPITTEL_3#KAPITTEL_3) gir rett til å bruke samisk i kontakt med offentlige organer i forvaltningsområdet. Møter en borgerrettet løsning samisk input og svarer flytende feil, er en rettighetssak, ikke en kvalitetssak. Minstekravet til dere er å vite hva som skjer når modellen får input den ikke behersker: å feile høyt og eskalere er forsvarlig; å feile flytende er det ikke.

### Tilgjengelighet og sårbare brukere

Tilgjengelighet ligger mest i løsningen--grensesnitt, struktur, skjermleserflyt--og dekkes av [forskrift om universell utforming av IKT-løsninger](https://lovdata.no/dokument/SF/forskrift/2013-06-21-732) uansett hva som ligger under panseret. Men noen egenskaper sitter i selve modellen, og de kan testes ved modellvalg:

- **Klarspråk på bestilling.** Klarer modellen konsekvent å produsere norsk på et gitt lesenivå, eller glir den tilbake til kansellispråk?
- **Robusthet mot ikke-standard input.** Mange av Navs brukere skriver ikke normert bokmål, på grunn av dysleksi, norsk som andrespråk, eller fordi teksten kommer via talegjenkjenning. Faller kvaliteten markant for disse brukerne, har dere en skjevhet som rammer akkurat dem som trenger tjenesten mest.

Dette området er nesten aldri benchmarket. Det betyr ikke at det er valgfritt, men at det havner i den lokale evalueringen.

### Sannferdighet, kalibrering og medgjørlighet

For veiledningsløsninger er den farligste feilen sjelden ren faktafeil, det er *medgjørlighet*: at modellen godtar brukerens uriktige premiss i stedet for å korrigere det. «Siden jeg har rett på AAP, kan jeg ...»  En modell som svarer hjelpsomt videre på det premisset, har produsert veiledning brukeren kan handle på og tape på. Test eksplisitt med oppgaver der premisset er feil, og se om modellen retter eller føyer seg. Generelle sannferdighetbenchmarker som TruthfulQA fanger ikke dette.

### Energi og ressursbruk

Den praktiske regelen: ikke bruk en større eller mer energikrevende modellklasse enn oppgaven krever. [AI Energy Score v2](https://huggingface.co/spaces/AIEnergyScore/Leaderboard) fant at resonneringsmodeller i snitt brukte rundt 30 ganger mer energi enn modeller uten resonnering...og i verste fall flere hundre ganger mer. Det betyr ikke at resonneringsmodeller aldri skal brukes, men heller at de bør reserveres for oppgaver som faktisk krever stegvis problemløsning. For mange oppgaver er en mindre modell, en klassisk ML-modell eller bedre informasjonsarkitektur både mer robust og mer bærekraftig.

Vit også om asymmetrien i datagrunnlaget: standardiserte energimål finnes i hovedsak for åpne modeller, fordi målingen krever tilgang til vektene. For de proprietære API-modellene de fleste team faktisk vurderer, er dere henvist til leverandørenes selvrapporterte tall, som har ulik metodikk og ulik systemgrense, og dermed begrenset sammenlignbarhet. Manglende sammenlignbare miljødata er et funn. Dokumentér det.

### Det som _ikke_ er teamets bord

Arbeidsforhold i leverandørkjeden og opprinnelsen til treningsdata er reelle ansvarlig-KI-temaer, men de hører hjemme i anskaffelse og leverandøroppfølging, ikke i produktteamets modellvalg. Dersom hvert team gjorde dette selv, ville det bety hundre versjoner av samme ubekreftede svar, om de samme få leverandørene. Lurer dere på status for en leverandør, spør innkjøp eller Innsikt & KI i stedet for å utrede det selv.

## Røde flagg

Siste sjekk før benchmarkresultater brukes som beslutningsgrunnlag:

| Spørsmål | Rødt flagg |
|---|---|
| Riktig modell? | Resultatet gjelder en annen modell, versjon eller API-konfigurasjon |
| Riktig systemtype? | Testen gjelder grunnmodellen, men dere bygger RAG, agent eller bruker guardrails |
| Riktig språk? | Resultatet er fra engelsk, men løsningen skal brukes på norsk |
| Riktig oppgave? | Benchmarken måler generell chat, men løsningen gjør saksstøtte, klassifisering eller retrieval |
| Riktig risiko? | Sikkerhet er testet, men ikke skjevhet, personvern eller kildeforankring |
| Uavhengig nok? | Resultatet bygger bare på leverandørens egne tall og metode |
| Ferskt nok? | Modellversjon, dato eller evalueringsoppsett er uklart |
| Brukt riktig? | Benchmarken brukes som om den dokumenterer samsvar med KI-forordningen |

Ett eller flere røde flagg betyr ikke at benchmarken er ubrukelig, det betyr at den et svakt signal, ikke hovedbegrunnelse. Beslutningen må i så fall bæres av lokal evaluering, dokumenterte forbehold og risikoreduserende tiltak.

## Når denne veiledningen ikke er nok

Benchmarker og lokal evaluering på teamnivå er utilstrekkelig når løsningen påvirker tilgang til ytelser, tjenester eller arbeid, brukes i saksbehandling eller beslutningsstøtte, prioriterer personer eller saker, eller møter innbyggere i sårbare situasjoner uten fagperson imellom. Da må modellvalget inngå i en bredere vurdering av hele systemet--formål, dataflyt, menneskelig kontroll, klage og overprøving, personvern og juridisk grunnlag--sammen med risikovurdering og juridisk vurdering.

Usikre på om dere er der? Ta kontakt for veiledning. Det er billigere å spørre tidlig.

## Kilderegister

Benchmarker, lenker og status per kilde vedlikeholdes separat i [kilderegisteret](./kilderegister-benchmarker.md), med dato og revisjonsansvar. Denne veiledningen skal kunne stå seg selv om enkeltkilder kommer og går.
