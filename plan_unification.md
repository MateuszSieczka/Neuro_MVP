# Plan: Faza U — Unifikacja przez Predictive Coding

> **Substrat skalowania ku neuro-AGI.** Ten dokument jest samodzielny —
> ignoruje `plan.md` i `new_plan.md`. Opisuje JEDNĄ fazę, która zamienia
> obecny system (zoo regionów + zoo reguł uczenia, ręcznie okablowane,
> zamrożone projekcje, node-perturbation w M1) na jeden skalowalny substrat:
> **jeden mikroukład, jedna reguła uczenia, jeden cel — minimalizacja
> swobodnej energii — na rozrastającym się grafie.**
>
> Cel fazy: usunąć dwie ściany skalowania zidentyfikowane w analizie:
> (1) struktura ustalona ręcznie, (2) słabe, źle skalujące się przypisanie
> zasługi (credit assignment). Patrz `colab/DIAGNOZA_phase6b.md` dla punktu
> wyjścia.

---

## 1. Teza

Mózg nie używa osobnej reguły na region. Używa **jednej zasady** —
minimalizacji błędu predykcji ważonego precyzją (predictive coding; Rao &
Ballard 1999) — zrealizowanej w **jednym, powtarzalnym mikroukładzie korowym**
(Mountcastle 1997; Bastos et al. 2012 "canonical microcircuit for predictive
coding"). Percepcja, uczenie, akcja i uwaga to ta sama operacja na różnych
zmiennych (Friston 2010 Free Energy Principle).

Obecny kod ma **wszystkie prymitywy** tej zasady, ale rozproszone i pod różnymi
regułami. Faza U je **unifikuje**: zostawia biologiczny prior makro-architektury
(jakie regiony, ich rola), ale zastępuje mikro-reguły jedną, a ręczną sekwencję
przepływu — relaksacją swobodnej energii na grafie, który **sam się rozrasta i
okablowuje**.

---

## 2. Decyzja: który predictive coding

**Wybór: hierarchiczne Predictive Coding (Rao-Ballard) jako substrat
inferencji i uczenia, zunifikowane z Active Inference (Friston FEP) dla akcji,
zrealizowane jako Predictive Coding Graphs (Salvatori et al. 2022) o DOWOLNEJ
topologii, z inferencją inkrementalną (Incremental PC; Salvatori, Millidge et
al. 2024) dla stabilności i szybkości, oraz precyzją jako uwagą (Feldman &
Friston 2010).**

Uzasadnienie (każdy punkt to odpowiedź na konkretne ograniczenie skalowania):

1. **Moc przypisania zasługi bez backpropu.** Whittington & Bogacz (2017)
   dowodzą, że PC **przybliża backpropagation** na dowolnym grafie obliczeniowym,
   przy CZYSTO LOKALNYCH aktualizacjach Hebbowskich (`Δw ∝ ε · aktywność`). To
   zdejmuje ścianę z M1: node-perturbation REINFORCE ma wariancję gradientu
   rosnącą z liczbą parametrów (Fiete & Seung 2006) — im większy system, tym
   gorzej. PC nie ma tego problemu: jego kredyt jest tak głęboki jak backprop,
   a pozostaje lokalny i biologiczny. **To jest właściwa zmiana skalująca.**

2. **Dowolna, rozrastalna topologia.** Salvatori et al. (2022, "Learning on
   Arbitrary Graph Topologies via Predictive Coding") pokazują, że PC uczy się
   na **dowolnym** grafie — nie potrzeba ręcznie projektowanego, jednokierunkowego
   dataflow. To bezpośrednio rozwiązuje problem ręcznego okablowania: dodanie
   regionu = dodanie węzłów/krawędzi do grafu PC; reguła się nie zmienia.

3. **Jeden cel obejmuje percepcję + uczenie + akcję + uwagę.** Active Inference
   (Friston 2010, 2017) rozszerza PC o działanie: akcja minimalizuje
   **oczekiwaną** swobodną energię (Expected Free Energy, EFE). Percepcja
   aktualizuje stany, uczenie aktualizuje wagi, akcja aktualizuje świat — ta
   sama zasada. Precyzja (Π) to uwaga i adaptacyjny learning-rate. To unifikuje
   to, co dziś jest osobnymi modułami (krytyk, aktor, neuromodulatory, attention).

4. **Stabilność i szybkość na poziomie systemu.** Klasyczne PC wymaga relaksacji
   stanów do równowagi na każde wejście (kosztowne, niestabilne przy
   ucieleśnieniu w czasie rzeczywistym). **Incremental PC** (Salvatori &
   Millidge 2024) i **dynamic/generalized PC** (predykcja NASTĘPNEGO stanu,
   nie relaksacja do statycznej równowagi) dają stabilną, jedno- lub
   kilku-krokową inferencję na cykl — pasuje do pętli sensomotorycznej online.

5. **Biologiczna wierność i zgodność z tezą projektu.** Rao-Ballard mapuje się
   na laminarną korę (Bastos et al. 2012: L2/3 = stany/reprezentacja, L4 =
   wejście/błąd, L5/6 = predykcja w dół). Friston FEP to wiodąca teoria
   obliczeniowa kory. Zero backpropu, tylko lokalne reguły. Projekt zostaje sobą.

**Czego NIE wybieramy i dlaczego:**
- *Czysty backprop / surrogate gradients* — łamie inwariant projektu (lokalność),
  nie-biologiczny.
- *e-prop (Bellec 2020)* — świetny do kredytu CZASOWEGO w rekurencyjnych SNN,
  ale to mechanizm uzupełniający, nie zasada całościowa. Używamy jego idei
  (śladów kwalifikowalności) dla wolnego, czasowego komponentu PC (patrz §6),
  ale rdzeniem jest PC/AIF.
- *Czysta klasyczna relaksacja Rao-Ballard* — zbyt wolna/niestabilna online;
  stąd wariant inkrementalny + dynamiczny.

---

## 3. Architektura docelowa — pięć zmian jako JEDNA faza

### U.1 — Jeden kanoniczny mikroukład PC (`core/pc_module.py`)

Definiujemy JEDEN moduł, z którego zbudowany jest każdy region:

```
PCModule:
    state μ        # reprezentacja (rate, low-pass)  — odpowiednik L2/3
    error ε        # błąd predykcji (rate)            — odpowiednik L4
    precision Π    # ważenie błędu (gain, uczone)     — Feldman & Friston 2010
    w_gen          # wagi generatywne (predykcja w dół do źródeł wejścia)
    lateral / kWTA # rzadkość + normalizacja (już w sparse.py / kWTA móżdżku)
```

Operacje (wszystkie lokalne):
- **predykcja:** `pred = w_gen @ μ` (do węzłów-dzieci w grafie),
- **błąd:** `ε = Π · (wejście − pred)`,
- **inferencja stanu:** `μ ← μ + η_μ · (Wᵀ ε_dół − ε_góra)` (gradient na
  swobodnej energii; jedno/kilka kroków na cykl — Incremental PC),
- **uczenie wag:** `Δw_gen ∝ ε · μ` (Hebbowskie, lokalne),
- **uczenie precyzji:** `Π ← f(EMA(ε²))` (precyzja = odwrotność wariancji
  błędu; neuromodulatory ją modulują — §U.5).

Region = `PCModule` (lub stos) + jego krawędzie w grafie. Różnice między
regionami to **łączność i priory**, nie inny kod. **Skalowanie = instancjonuj
więcej kopii.** Punkt zaczepienia w kodzie: `core/error_neuron.py` JUŻ jest tym
modułem (populacje L4-error + L2/3-state, `en_prediction_error_rate`,
`en_belief`) — staje się rdzeniem; resztę przepisujemy na jego bazie.

### U.2 — Jedna reguła uczenia (PC zastępuje zoo reguł)

Dziś: STDP w korze, Marr-Albus w móżdżku, node-perturbation w M1,
three-factor w BG, TD w VTA, one-shot w HC — **sześć różnych reguł**, nie
skalują się jednolicie i każda ma własne stałe.

Po U.2: **jedna reguła** — lokalne, precyzją-ważone uczenie Hebbowskie błędu
predykcji (`Δw ∝ Π·ε·μ`) na każdym module. Konsekwencje per-region:
- **Kora:** Rao-Ballard na laminach (Bastos 2012). STDP znika jako osobna reguła
  — staje się czasową realizacją PC (Whittington & Bogacz 2017 pokazują związek
  STDP↔PC). `core/cortex.py` reinterpretowany, nie wyrzucony.
- **Móżdżek:** adaptacyjny filtr Dean-Porrill JUŻ jest PC modelu przedniego
  (predykcja proprio, błąd pnący). Zostaje, spięty jako podgraf PC. (Naprawiony
  w bieżącej rundzie — patrz DIAGNOZA §3.1/3.2.)
- **World-model:** `core/world_model.py` JUŻ jest generatywnym PC
  (`error_neuron` enkoder + dekoder). Przestaje być specjalny — to węzeł grafu.
- **VTA/nagroda:** RPE to błąd predykcji na generatywnym modelu nagrody;
  dopamina = **precyzja** predykcji nagrody (FitzGerald, Dolan & Friston 2015).
  TD znika jako osobna reguła — to PC na osi wartości.
- **BG:** wybór akcji = precyzja polityk (Friston 2017 "BG as policy
  precision"), nie osobny aktor-krytyk. Patrz §U.5.
- **HC/sen:** kodowanie epizodyczne i replay to **generatywny replay**
  minimalizujący swobodną energię offline (sen = offline FEP; konsolidacja =
  destylacja generatywna). `sleep_replay.py` reinterpretowany.

### U.3 — Graf PC o dowolnej topologii + relaksacja zamiast ręcznej sekwencji

Dziś `action_brain_cognitive_step` to **ręcznie zakodowana sekwencja** wywołań
regionów w ustalonej kolejności. To jest ręczne okablowanie dataflow.

Po U.3: stan mózgu to **graf modułów PC** (`PCGraph`). Jeden cykl poznawczy =
**N kroków relaksacji** swobodnej energii po grafie (Salvatori 2022: PC działa
na dowolnym grafie; Incremental PC: N małe, np. 1-4). Kolejność wyłania się z
przepływu błędów, nie z ręcznego skryptu. Nowy region = nowe węzły + krawędzie
w grafie; `cognitive_step` się nie zmienia.

`core/free_energy.py` dostaje wreszcie konsumenta: globalna swobodna energia
grafu = suma `Π·ε²` po modułach; to wielkość, którą wszystko minimalizuje
(percepcja, uczenie) i którą diagnozujemy.

### U.4 — Plastyczne wagi dalekozasięgowe + plastyczność strukturalna (samo-okablowanie)

- **Koniec zamrożonych losowych projekcji.** `w_l5_ct`, `w_l5_mossy`,
  `w_efference_mossy` (i wszystkie krawędzie grafu) to teraz wagi generatywne PC
  → uczą się tą samą regułą (`Δw ∝ Π·ε·μ`). Routing nie zależy już od trafności
  losowego inicju. (`w_dn_motor` móżdżku zostaje stałe — to świadomie stałe
  wagi wyjściowe filtra adaptacyjnego, Dean & Porrill; wyjątek uzasadniony.)
- **Plastyczność strukturalna jako redukcja swobodnej energii.** `core/sparse.py`
  (`synaptogenesis`, `prune_below`) dostaje konsumenta: krawędzie/węzły **rosną**,
  gdy redukują przewidywalny błąd (free energy), i są **przycinane** wg kosztu
  okablowania (Chklovskii 2004; Rakic 1988 przycinanie rozwojowe). Kryterium jest
  to samo, co wszędzie — swobodna energia + koszt. **Graf sam się okablowuje i
  rozrasta z doświadczenia**, zamiast być ustalony w czasie projektu.

### U.5 — Akcja = active inference (EFE), hierarchia celów, ciekawość jako epistemic value

- **Komenda ruchowa to predykcja, nie polityka.** Active inference: M1 przewiduje
  pożądany stan proprioceptywny; ciało jest napędzane, by spełnić predykcję
  (Adams, Shipp & Friston 2013 "predictions not commands"; łuki odruchowe
  rdzenia minimalizują błąd proprio). To zastępuje node-perturbation REINFORCE
  w `core/m1.py` — usuwa źle-skalujący się estymator. Móżdżek (model przedni)
  dostarcza predykcji konsekwencji; rdzeń/ciało domyka pętlę.
- **Wybór akcji = argmin oczekiwanej swobodnej energii (EFE).** Wreszcie konsumuje
  `free_energy.expected_free_energy`. EFE rozkłada się na:
  - **pragmatyczną** (osiągnij preferowane stany = cele/nagroda),
  - **epistemiczną** (information gain = ciekawość).
  To unifikuje cel i ciekawość w JEDNYM równaniu. `wm_learning_progress`
  (Oudeyer) staje się przybliżeniem składnika epistemicznego.
- **Hierarchia celów za darmo.** Hierarchiczne PC nad stanami = hierarchiczne
  active inference nad celami (Friston, Pezzulo et al. 2018 "hierarchical active
  inference"): wyższe moduły przewidują abstrakcyjne cele, niższe je realizują
  ruchowo. To jest brakująca otwartość/kompozycyjność (lever 5) — nie osobny
  mechanizm, lecz głębokość tego samego grafu.
- **Neuromodulatory = kontrolery precyzji.** ACh = precyzja sensoryczna, NE =
  zmienność/volatility → tempo uczenia, DA = precyzja polityki/nagrody, 5-HT =
  horyzont. (Yu & Dayan 2005; FitzGerald 2015; Parr & Friston 2017.) `core/
  neuromodulator.py` reinterpretowany jako bus precyzji — częściowo już tym jest.

---

## 4. Co znika, co zostaje

**Znika (zastąpione jedną zasadą):**
- Zoo reguł uczenia jako OSOBNE mechanizmy (STDP, Marr-Albus, three-factor, TD,
  one-shot, node-perturbation) — stają się realizacjami PC/AIF.
- Ręcznie zakodowana sekwencja przepływu w `cognitive_step` → relaksacja na grafie.
- Zamrożone losowe projekcje dalekozasięgowe → uczone wagi generatywne.
- node-perturbation REINFORCE w M1 → active inference.

**Zostaje (to są dobre priory, nie bugi):**
- Biologiczny prior makro-architektury (jakie regiony, role) — to genetyczny
  blueprint, sens projektu. PC działa NA nim, nie zamiast.
- Neuromodulatory, sen/replay, oscylacje, ucieleśnienie (MJX) — reinterpretowane
  w ramach FEP, nie usunięte.
- Móżdżek jako adaptacyjny filtr/model przedni (już ≈ PC).

---

## 5. Mapowanie na istniejący kod

| Moduł dziś | Rola w Fazie U |
|------------|----------------|
| `core/error_neuron.py` | **rdzeń** — kanoniczny `PCModule` (μ/ε/Π) |
| `core/cortex.py` | stos PC na laminach (Bastos 2012); STDP → realizacja PC |
| `core/world_model.py` | węzeł grafu (generatywny PC); przestaje być specjalny |
| `core/cerebellum.py` | podgraf modelu przedniego (już adaptacyjny filtr) |
| `core/precision_bus.py` | precyzja Π wszędzie = uwaga + adaptacyjny lr |
| `core/free_energy.py` | **konsument** — globalna FE (percepcja/uczenie) + EFE (akcja) |
| `core/basal_ganglia.py` | precyzja polityk (Friston 2017), nie osobny aktor-krytyk |
| `core/vta.py` | błąd predykcji nagrody; DA = precyzja (FitzGerald 2015) |
| `core/m1.py` | active inference: komenda = predykcja proprio (Adams 2013) |
| `core/neuromodulator.py` | bus precyzji (ACh/NE/DA/5-HT) |
| `core/hippocampus.py`, `sleep_replay.py` | generatywny replay = offline FEP |
| `core/sparse.py` | wzrost/przycinanie grafu wg redukcji FE + koszt okablowania |
| `core/brain_graph.py` | `PCGraph` + pętla relaksacji zamiast ręcznej sekwencji |

Wniosek: **kod już zawiera ~80% prymitywów PC** (error_neuron, world_model,
precision_bus, free_energy, kWTA, neuromodulatory). Faza U nie jest budową od
zera — to **unifikacja i scalenie** pod jedną zasadą + jeden substrat.

---

## 6. Kredyt czasowy (uzupełnienie)

PC daje głęboki kredyt PRZESTRZENNY. Dla zadań ROZCIĄGNIĘTYCH w czasie
(sekwencje, opóźniona nagroda) dokładamy:
- **Dynamic / generalized predictive coding** — moduły przewidują NASTĘPNY stan
  (predykcja w czasie), nie tylko hierarchię w przestrzeni. `world_model` już to
  robi; uogólniamy na cały graf (uogólnione współrzędne, Friston 2008).
- **Ślady kwalifikowalności** (idea z e-prop, Bellec 2020; Yagishita 2014 okno
  reinforcementu 0.3-2 s) na wolnych krawędziach — most między zdarzeniem a
  opóźnionym błędem/nagrodą. Bufor `delay_*` w kodzie już realizuje opóźnienia.

---

## 7. Dlaczego to skaluje (jawnie)

1. **Credit assignment:** PC ≈ backprop (Whittington & Bogacz 2017) → kredyt
   nie degraduje się z rozmiarem, jak w node-perturbation. Ściana #2 zdjęta.
2. **Struktura:** graf rośnie/przycina się sam (U.4) na dowolnej topologii
   (Salvatori 2022) → skalowanie = replikuj mikroukład + pozwól strukturze
   wyłonić się. Ściana #1 zdjęta.
3. **Cel:** jedno równanie (FE/EFE) napędza percepcję, uczenie, akcję, uwagę i
   eksplorację → brak osobnych, ręcznie zszywanych mechanizmów na każdą zdolność;
   nowe zdolności = głębsza hierarchia, nie nowy kod.

To jest spójna (wciąż NIEUDOWODNIONA empirycznie na skali) ścieżka do
neuro-AGI: jeden substrat, jedna zasada, rosnący graf.

---

## 8. Ryzyka i otwarte problemy (uczciwie)

1. **PC ≈ backprop tylko pod założeniami** (Whittington-Bogacz wymaga pewnej
   struktury i relaksacji; Incremental PC robi przybliżenie). W praktyce na dużą
   skalę PC bywa wolniejszy/trudniejszy do strojenia niż backprop — to wciąż
   aktywny problem badawczy (2022-2025).
2. **Stabilność relaksacji online.** Zbyt mało kroków relaksacji = zła
   inferencja; za dużo = wolno i niestabilnie w pętli ucieleśnionej. Trzeba
   wyważyć (Incremental/dynamic PC to adresuje, ale nie rozwiązuje w 100%).
3. **Plastyczność strukturalna jest niestabilna** — wzrost/przycinanie sterowane
   FE może się rozjechać (eksplozja krawędzi lub zapadnięcie). Potrzeba twardego
   kosztu okablowania i homeostazy.
4. **Active inference dla bogatej motoryki** — "predictions not commands" działa
   w prostych przypadkach; dla zręcznej kontroli to wciąż otwarte (planowanie EFE
   jest kosztowne obliczeniowo — wybór polityk rośnie wykładniczo bez przycinania).
5. **To duża przebudowa.** Ryzyko regresji wszystkiego naraz. Mityguj:
   buduj `PCModule` + `PCGraph` obok, migruj region po regionie z testem
   równoważności, aż `cognitive_step` stanie się relaksacją grafu.

---

## 9. Kryteria sukcesu — co testować (zdolności, nie hydraulika)

Faza U jest sukcesem, jeśli (mierzone na rozszerzeniu `phase6b_capability.ipynb`):
1. **Unifikacja:** jeden `PCModule` + jedna reguła budują wszystkie regiony;
   liczba osobnych reguł uczenia spada do 1 (+ stałe wyjścia móżdżku).
2. **Skalowanie kredytu:** głęboka hierarchia (≥4 poziomy) uczy się zadania,
   którego node-perturbation NIE uczył się (np. predykcja wielokrokowa) — dowód,
   że PC daje kredyt, którego perturbacja nie dawała.
3. **Samo-okablowanie:** po włączeniu plastyczności strukturalnej graf zmienia
   łączność i poprawia FE bez ręcznego dodawania krawędzi.
4. **Active inference:** reach działa BEZ node-perturbation — komenda jako
   predykcja + EFE; ciekawość = składnik epistemiczny EFE (nie osobny bonus).
5. **Globalna swobodna energia maleje** w trakcie życia agenta na held-out
   doświadczeniu — jedna liczba, jeden cel, mierzalny.
6. **Brak regresji** zdolności sensomotorycznych z `phase6b_capability.ipynb`
   (C1-C6) po migracji.

---

## 10. Sedno

Obecny system to dobry biologiczny PROTOTYP z właściwą filozofią, ale ze
strukturą i regułami ustalonymi ręcznie — stąd plateau. Faza U zamienia go w
**jeden substrat, jedną zasadę (predictive coding / active inference), jeden
rosnący graf**. To jedyna z rozważanych zmian, która rusza WIĄŻĄCE ograniczenie
(siła i skalowalność uczenia), nie tylko sztywność struktury. Prymitywy już są
w kodzie. To nie gwarancja AGI — to przejście z „ręcznie zaprojektowany mózg"
na „mózg, który dostraja, rozrasta i okablowuje się sam z doświadczenia", czyli
jedyna rodzina podejść, która w ogóle ma szansę skalować się w tym kierunku.
