# Plan: Faza U — Unifikacja przez Predictive Coding

> **Substrat skalowania ku neuro-AGI.** Ten dokument jest samodzielny —
> ignoruje `plan.md` i `new_plan.md`. Opisuje JEDNĄ fazę, która zamienia
> obecny system (zoo regionów + zoo reguł uczenia, ręcznie okablowane,
> zamrożone projekcje, node-perturbation w M1) na jeden skalowalny substrat:
> **jeden mikroukład, jedna reguła uczenia, jeden cel — minimalizacja
> swobodnej energii — na rozrastającym się grafie.**
>
> **To NIE jest plan reinterpretacji.** Każdy punkt U.X to konkretne
> przepisanie kodu: nowe pola stanu, nowe równania, nowe call-site'y,
> test równoważności / nie-regresji. Tam, gdzie poprzednia wersja tego
> dokumentu pisała „region X reinterpretowany jako PC", ta wersja podaje
> regułę, którą zastępujemy starą, i mechanizm migracji. Celem jest AGI;
> przyjmujemy ryzyko dużej, jednoczesnej przebudowy reguł uczenia i
> mitygujemy je migracją region-po-regionie z testami (§8, §9).
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

Obecny kod ma **wiele prymitywów** tej zasady, ale rozproszone, pod różnymi
regułami i — co kluczowe — **bez prawdziwej dynamiki inferencyjnej PC**.
`core/error_neuron.py` ma populacje stanu (μ) i błędu (ε), lecz jego
„inferencja" to integracja spike'ów AdEx, nie relaksacja swobodnej energii
`μ̇ = −∂F/∂μ`, od której zależy równoważność PC↔backprop (Whittington &
Bogacz 2017). Faza U **nie przekleja etykiety** na ten moduł — buduje
kanoniczny moduł rate z jawną relaksacją (U.1), przepisuje każdą regułę
regionu na jedną regułę PC (U.2), zastępuje ręczną sekwencję relaksacją na
grafie (U.3), uplastycznia okablowanie (U.4) i czyni akcję active inference
(U.5).

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
   przy CZYSTO LOKALNYCH aktualizacjach Hebbowskich (`Δw ∝ ε · aktywność`),
   ALE TYLKO gdy stany μ są **relaksowane do (przybliżonego) punktu stałego
   swobodnej energii** przed aktualizacją wag. To jest dokładnie ta dynamika,
   której obecny kod nie ma i którą U.1 wprowadza. Bez niej nie ma korzyści
   skalowania — jest tylko zmiana nazwy. Z nią: kredyt tak głęboki jak backprop,
   lokalny i biologiczny. To zdejmuje ścianę z M1 (node-perturbation REINFORCE
   ma wariancję gradientu rosnącą z liczbą parametrów — Fiete & Seung 2006).

2. **Dowolna, rozrastalna topologia.** Salvatori et al. (2022, "Learning on
   Arbitrary Graph Topologies via Predictive Coding") pokazują, że PC uczy się
   na **dowolnym** grafie — nie potrzeba ręcznie projektowanego, jednokierunkowego
   dataflow. Dodanie regionu = dodanie węzłów/krawędzi do grafu PC; reguła się
   nie zmienia.

3. **Jeden cel obejmuje percepcję + uczenie + akcję + uwagę.** Active Inference
   (Friston 2010, 2017) rozszerza PC o działanie: akcja minimalizuje
   **oczekiwaną** swobodną energię (Expected Free Energy, EFE). Percepcja
   aktualizuje stany, uczenie aktualizuje wagi, akcja aktualizuje świat — ta
   sama zasada. Precyzja (Π) to uwaga i adaptacyjny learning-rate.

4. **Stabilność i szybkość na poziomie systemu.** Klasyczne PC wymaga relaksacji
   stanów do równowagi na każde wejście. **Incremental PC** (Salvatori &
   Millidge 2024) i **dynamic/generalized PC** (predykcja NASTĘPNEGO stanu)
   dają stabilną, jedno- lub kilku-krokową inferencję na cykl — pasuje do pętli
   sensomotorycznej online. U.1 implementuje wariant inkrementalny (N=1–4 kroki
   relaksacji na cykl).

5. **Biologiczna wierność i zgodność z tezą projektu.** Rao-Ballard mapuje się
   na laminarną korę (Bastos et al. 2012: L2/3 = stany μ, L4 = błąd ε, L5/6 =
   predykcja w dół). Zero backpropu, tylko lokalne reguły.

**Czego NIE wybieramy i dlaczego:**
- *Czysty backprop / surrogate gradients* — łamie inwariant projektu (lokalność).
- *e-prop (Bellec 2020)* — używamy jego idei (śladów kwalifikowalności) dla
  wolnego, czasowego komponentu (§6), ale rdzeniem jest PC/AIF.
- *Czysta klasyczna relaksacja Rao-Ballard* — zbyt wolna/niestabilna online;
  stąd wariant inkrementalny + dynamiczny.

---

## 3. Architektura docelowa — pięć przepisań jako JEDNA faza

Notacja wspólna dla całego §3. Moduł `m` ma:
- `μ` — stan/przekonanie (rate, low-pass), wektor `n_state`;
- `ε` — błąd predykcji na interfejsie wejścia modułu, wektor `n_obs`;
- `Π` — precyzja (diagonalna, uczona), wektor `n_obs`;
- `w_gen` — wagi generatywne (predykcja w dół: `pred = w_genᵀ μ`);
- opcjonalnie `w_amort` — szybka, amortyzowana inicjalizacja μ (Tscshantz 2023).

Lokalna swobodna energia modułu: `F_m = ½ Σ Π ⊙ ε²`, `ε = obs − g(w_genᵀ μ)`,
gdzie `obs` to predykcje przychodzące od węzłów-rodziców / sygnał sensoryczny,
a `g` to funkcja aktywacji (domyślnie tożsamość lub `tanh`).

### U.1 — Kanoniczny mikroukład PC z PRAWDZIWĄ relaksacją (`core/pc_module.py`)

**Co budujemy (nowy plik, obok `error_neuron.py`).** Rate-mode `PCModule` z
jawną dynamiką inferencyjną. Rate, nie spike, bo relaksacja PC wymaga kilku
szybkich iteracji μ na cykl — czego spike'owa integracja AdEx nie realizuje.
`error_neuron.py` zostaje (kora spike'owa go używa, `cortex.py:86,270`), ale
przestaje być „rdzeniem PC"; rdzeniem staje się `PCModule`.

Stan (`PCModuleState`, `eqx.Module`):
```
mu        # (n_state,)  belief μ
w_gen     # (n_obs, n_state)  generative; pred = w_gen @ mu      ← UCZONE
w_amort   # (n_obs, n_state)  optional feedforward init of mu    ← UCZONE
pi        # (n_obs,)  precision = 1/(var_ε + floor)             ← UCZONE
pe_var    # (n_obs,)  EMA wariancji ε (do Π; wektorowy PrecisionChannel)
```

Operacje (wszystkie lokalne; `pc_relax`, `pc_learn`, `pc_predict`):

1. **Predykcja w dół (do dzieci / źródeł wejścia):**
   `pred = g(w_gen @ mu)`.
2. **Błąd ważony precyzją:** `ε = obs − pred`, `ξ = pi ⊙ ε` (Friston 2010 §3.2).
3. **Inferencja stanu (RELAKSACJA — to jest nowość):**
   `mu ← mu + η_μ · ( (w_gen ⊙ g'(·))ᵀ ξ  −  ξ_up  −  λ·mu )`,
   iterowane `N` kroków (Incremental PC; N domyślnie 2). `ξ_up` to błąd
   ważony precyzją przekazany w GÓRĘ do tego modułu przez krawędzie wychodzące
   (w grafie U.3); dla węzła liścia `ξ_up = 0`. `λ` to mały wyciek (prior μ→0,
   Rao-Ballard prior gaussowski).
4. **Uczenie wag (PO relaksacji, Hebbowskie):**
   `Δw_gen = η_w · ξ ⊗ g(mu)`  — to dokładnie `−∂F/∂w_gen` (Whittington &
   Bogacz 2017, eq. 8). `Δw_amort = η_a · (mu − w_amortᵀ obs) ⊗ obs` (regresja
   amortyzacyjna, opcjonalnie).
5. **Uczenie precyzji:** `pe_var ← EMA(ε²)`, `pi = 1/(pe_var + floor)`
   (FitzGerald, Dolan & Friston 2015; neuromodulatory skalują `pi` — §U.5).
   Reużywa wektorowej wersji `precision_bus.PrecisionChannel` (Welford EMA,
   już zaimplementowana, `precision_bus.py:118`).

**Punkt zaczepienia / równoważność.** `error_neuron.py` dostarcza wzorca pól
(μ=`state_rate`, ε=`error_rate`, trójki `w_in/w_bu/w_td`), ale jego `en_step`
(`error_neuron.py:263-291`) integruje spike'i — **nie relaksuje F**. `PCModule`
implementuje pkt 3 explicite. To realna zmiana mechanizmu, nie rename.

**Test U.1 (warunek wejścia do dalszych kroków).** ZREALIZOWANE w
`core/pc_module.py` + `tests/test_phaseu_pc_module.py` (4/4 zielone):
- (a) *Zbieżność relaksacji:* `pc_relax` monotonicznie redukuje `F_m` i dochodzi
  do punktu stałego (residual `max|∂F/∂μ| < 1e-3`). Potwierdzone.
- (b) *PC = backprop (twarda bramka):* na 3-warstwowym `PCModule` cosinus między
  lokalnym gradientem PC a `jax.grad` (backprop) wynosi **1.0000** (liniowy i
  tanh, wszystkie seedy), gdy używamy **równowagi fixed-prediction** (predykcje/
  presynapsy trzymane na feedforward podczas relaksacji błędów; Millidge &
  Bogacz 2020; Song 2020 Z-IL). To dynamika relaksacji odtwarza backprop, nie
  ręczny backprop. **Dowód, że zdobyliśmy credit assignment, nie tylko nazwę.**
- **Ważne odkrycie:** *standardowa* relaksacja online (predykcje przeliczane
  co krok — tryb biologicznie wierny, używany w grafie U.3) daje cosine ≈ 0.85–
  0.94, NIE 1.0, nawet w dokładnym punkcie stałym i przy aktywacji liniowej. To
  nie błąd — to udokumentowana własność relaksowanego PC (używa zrelaksowanego μ
  jako czynnika presynaptycznego; backprop używa feedforward). Ścisła
  równoważność wymaga fixed-prediction. Konsekwencja dla U.3: dla zadań
  wymagających kredytu na poziomie backpropu (głęboka hierarchia, §9.4) używamy
  trybu fixed-prediction; dla percepcji online wystarcza relaksacja zwykła.

### U.2 — Jedna reguła uczenia: PRZEPISANIE sześciu reguł na `Δw = η·Π·ε·μ`

Dziś realnie sześć reguł (DIAGNOZA + lektura kodu): STDP/anti-Hebb w korze
(`cortex.py`, `error_neuron.en_update_weights`), Marr-Albus LTD w móżdżku
(`cerebellum.py`), node-perturbation REINFORCE w M1 (`m1.py:328`), three-factor
aktor-krytyk w BG (`basal_ganglia.py`), TD(0) w VTA (`vta.py`), one-shot w HC
(`hippocampus.py`). Każda ma własne stałe i własny mechanizm.

Po U.2: **każdy region to `PCModule` (lub stos), a JEDYNA reguła plastyczności
to** `Δw_gen = η · (Π⊙ε) ⊗ g(μ)` **(plus uczenie Π).** Poniżej, per region,
co dokładnie usuwamy i czym zastępujemy — to są przepisania, nie reinterpretacje:

- **Kora (`cortex.py`).** L2/3 już jest `ErrorNeuron` PC (`cortex.py:86,270`),
  ale `w_l23_l5` i wagi L4/L5 uczą się osobnym Hebbianem/STDP. Przepisanie:
  laminy mapują się na JEDEN `PCModule` (Bastos 2012) — L4=ε, L2/3=μ,
  L5=projekcja generatywna `w_gen` w dół + odczyt. **Usuwamy** osobny STDP na
  `w_l23_l5`; staje się on `w_gen` uczonym regułą `Π·ε·μ`. Migracja: zastąp
  `cortical_area_step/_update` wywołaniem `pc_relax/pc_learn` na module
  korowym; `en_step` spike'owy zostaje tylko jeśli potrzebny do oscylacji,
  inaczej znika.
- **Móżdżek (`cerebellum.py`).** Już adaptacyjny filtr ≈ przedni model PC
  (Dean-Porrill). Przepisanie: granule = wejście (μ-źródło), Purkinje = błąd ε
  (włókna pnące = ε z przestrzeni stawów), `w_gp` = wagi generatywne uczone
  `Π·ε·μ` — TĄ SAMĄ regułą, zastępując kowariancyjną LTD Marr-Albus.
  **Wyjątek zachowany:** `w_dn_motor` (mikrostrefa→wyjście) zostaje STAŁE — to
  świadomie stałe wagi wyjściowe filtra (Dean & Porrill 2010), nie pominięcie
  reguły. Sygnał uczący już jest poprawnie sprzężony transpozycją
  (`brain_graph.py:1434`).
- **VTA / wartość (`vta.py`).** TD(0) `rpe = (r−b)+γV(s′)−V(s)` przepisane na
  **temporalny PC węzła wartości:** V = `μ` modułu nagrody, błąd nagrody
  `ε_r = r + γ·pred_V_next − pred_V` to błąd predykcji na krawędzi CZASOWEJ
  (§6), dopamina = `Π` predykcji nagrody (FitzGerald 2015). `Δw_value = η·Π·ε_r·μ`.
  Bootstrapping γV(s′) staje się krawędzią dynamiczną „przewiduj wartość
  następnego stanu", nie osobnym wzorem TD.
- **BG (`basal_ganglia.py`).** Aktor-krytyk przepisany na **precyzję polityk**
  (Friston 2017). Krytyk = węzeł wartości (wyżej). Aktor: polityka to prior nad
  akcjami ważony precyzją; wybór = argmin EFE (U.5); aktualizacja priorów
  `Δw_policy = η·Π_policy·ε_outcome·μ`. **Usuwamy** osobne reguły aktora i
  krytyka; zostaje jedna reguła na węzłach wartości i polityki.
- **M1 (`m1.py`).** Node-perturbation REINFORCE **usunięte w całości** (źle
  skalujący estymator). Zastąpione active inference w U.5.
- **HC / sen (`hippocampus.py`, `sleep_replay.py`).** One-shot kodowanie =
  szybka inferencja PC przy WYSOKIM Π (jeden krok relaksacji wystarcza do
  zapisu, McClelland 1995 fast weights). Replay = **generatywny replay**:
  próbkuj z modelu generatywnego offline, relaksuj graf, destyluj regułą
  `Π·ε·μ` (sen = offline FEP). Zastępuje osobną regułę one-shot regułą wspólną
  + trybem offline grafu (U.3).

**Test U.2 (per region, podczas migracji):** po przepięciu danego regionu na
`PCModule`+wspólną regułę, zdolności sensomotoryczne C1–C6 z
`phase6b_capability.ipynb` nie regresują o więcej niż margines tolerancji; a
licznik osobnych reguł plastyczności spada o jeden. Po całym U.2: licznik = 1
(+ stałe `w_dn_motor`).

### U.3 — Graf PC o dowolnej topologii + relaksacja zamiast ręcznej sekwencji (`core/pc_graph.py`)

**Co zastępujemy.** `action_brain_cognitive_step` (`brain_graph.py:1199`) to
ręcznie zakodowana sekwencja; `_perceive_substep` (`:920`) ma sztywny porządek
wywołań regionów. To jest ręczne okablowanie dataflow.

**Czym.** Stan mózgu to `PCGraph`: zbiór węzłów (`PCModule`) + krawędzie
(`w_gen` + bufor opóźnienia, reużyj `DelayBuffer`, `brain_graph.py:155`). Jeden
cykl poznawczy:
1. **Clamp:** węzły obserwacyjne ← `sensory`; węzły celu ← preferowane stany
   (priory dla active inference, U.5).
2. **Relaksacja:** `for n in range(N): ` każdy węzeł liczy ε z predykcji
   przychodzących i aktualizuje μ (`pc_relax`, Incremental PC, N=1–4).
   Kolejność wyłania się z przepływu błędów po krawędziach, nie ze skryptu.
3. **Uczenie:** po relaksacji `pc_learn` na każdej krawędzi (`Δw = η·Π·ε·μ`).
4. **Akcja:** odczyt węzłów akcji (U.5).

`core/free_energy.py` dostaje wreszcie konsumenta. Przywracamy
`variational_free_energy(precision, error) = ½ Σ Π·ε²` (usuniętą jako
„redundantny one-liner" — `free_energy.py:9-13`) i dodajemy
`graph_free_energy(graph) = Σ_nodes F_node`. To wielkość, którą wszystko
minimalizuje i którą diagnozujemy (kryterium §9.5).

**Co MUSI być zachowane jawnie (tu siedzi obecna poprawność).** Bieżąca
kolejność koduje przyczynowość: percepcja `s_{t+1}` → domknięcie poprzedniego
przejścia → akcja, z eligibility traces, celem TD i opóźnieniem 1-cyklowym.
Naiwna relaksacja to gubi. Mapowanie:
- **Predykcje czasowe** (następny stan/wartość) → krawędzie dynamiczne PC
  (§6), nie ręczna sekwencja.
- **Cel TD** → krawędź czasowa wartość(t)→wartość(t+1) (U.2 VTA).
- **Kredyt międzycyklowy** → ślady kwalifikowalności na wolnych krawędziach
  (§6), nie sztywny porządek wywołań.
Plan migracji (§szczegóły niżej): buduj `PCGraph` obok, dodawaj węzły po
jednym, utrzymuj stary `cognitive_step` aż graf odtworzy C1–C6.

### U.4 — Plastyczne wagi dalekozasięgowe + plastyczność strukturalna (samo-okablowanie)

- **Koniec zamrożonych projekcji.** `w_l5_ct`, `w_l5_mossy`, `w_efference_mossy`
  są dziś w `ActionBrainParams` (`brain_graph.py:402-408`) = de facto zamrożone.
  **Przenosimy je do stanu** (`ActionBrainState` / odpowiednich węzłów grafu)
  jako wagi generatywne `w_gen` i uczymy regułą `Π·ε·μ`. `w_dn_motor`
  (`:472`) zostaje stałe — wyjątek jak w U.2.
- **Plastyczność strukturalna jako redukcja swobodnej energii.** `core/sparse.py`
  (`synaptogenesis`, `prune_below`, `SparseConnectivity`) jest dziś **martwym
  kodem** (nieużywany w żywym obwodzie; tylko eksport + notebooki). Wpinamy go w
  graf:
  - **Wzrost:** krawędź/węzeł powstaje, gdy redukuje przewidywalny błąd —
    kryterium `ΔF < 0` w oknie po próbnym dodaniu (lub proxy: pre/post
    współaktywne ∧ wysokie `F_node`, czyli „jest co przewidywać").
  - **Przycinanie:** wg kosztu okablowania (Chklovskii 2004), `|w| < θ`
    skalowane długością/kosztem; przycinanie rozwojowe (Rakic 1988).
  - **Homeostaza (konieczna, §8.3):** twardy limit aktywnych krawędzi na węzeł +
    docelowa gęstość — inaczej eksplozja/zapadnięcie.
  **Kolejność:** to robimy OSTATNIE, po stabilnym U.1–U.3, bo jest najbardziej
  niestabilne.

**Test U.4:** (a) po uplastycznieniu projekcji globalna FE na held-out nie
rośnie (nie psujemy); (b) po włączeniu strukturalnej: graf zmienia łączność
i poprawia FE bez ręcznego dodawania krawędzi, przy stabilnej gęstości.

### U.5 — Akcja = active inference (EFE), hierarchia celów, ciekawość jako epistemic value

- **Komenda ruchowa to predykcja, nie polityka.** M1 (zastępując `m1.py`
  REINFORCE) to węzeł generatywny przewidujący pożądany stan proprioceptywny;
  ciało napędzane, by spełnić predykcję (Adams, Shipp & Friston 2013; łuki
  odruchowe rdzenia minimalizują proprio-ε). Móżdżek (model przedni) dostarcza
  predykcji konsekwencji; rdzeń/ciało domyka pętlę.
- **Wybór akcji = argmin EFE.** Wpinamy `expected_free_energy`
  (`free_energy.py:42`, dziś martwe, nawet poza `__all__`). EFE rozkłada się na:
  - **pragmatyczną** (osiągnij preferowane stany = cele/nagroda),
  - **epistemiczną** (information gain = ciekawość).
  `wm_learning_progress` (`world_model.py:469`, Oudeyer) staje się
  przybliżeniem składnika epistemicznego. Jedno równanie unifikuje cel i
  ciekawość.
- **Hierarchia celów za darmo.** Hierarchiczne PC nad stanami = hierarchiczne
  active inference nad celami (Friston, Pezzulo et al. 2018): wyższe węzły
  przewidują abstrakcyjne cele, niższe realizują ruchowo. Otwartość/
  kompozycyjność = głębokość tego samego grafu, nie osobny mechanizm.
- **Neuromodulatory = kontrolery precyzji.** `core/neuromodulator.py` przepisany
  na bus precyzji: ACh = `Π` sensoryczna, NE = volatility → tempo uczenia,
  DA = `Π` polityki/nagrody, 5-HT = horyzont (Yu & Dayan 2005; FitzGerald 2015;
  Parr & Friston 2017). Skalują `pi` w `PCModule` (U.1 pkt 5).

**Test U.5:** reach działa BEZ node-perturbation — komenda jako predykcja +
EFE; ablacja składnika epistemicznego mierzalnie zmienia eksplorację.

---

## 4. Co znika, co zostaje

**Znika (realnie usunięte z kodu, nie przemianowane):**
- node-perturbation REINFORCE w M1 (`m1.py:328`) → active inference (U.5).
- Osobny STDP na `w_l23_l5` w korze → `w_gen` regułą wspólną (U.2).
- Kowariancyjna LTD Marr-Albus jako osobna reguła → `Π·ε·μ` na `w_gp` (U.2).
- Wzór TD(0) w VTA → temporalny PC węzła wartości (U.2/§6).
- Osobne reguły aktora i krytyka w BG → precyzja polityk + węzeł wartości (U.2).
- Reguła one-shot w HC → szybka inferencja PC + generatywny replay (U.2).
- Ręczna sekwencja `cognitive_step`/`_perceive_substep` → relaksacja grafu (U.3).
- Zamrożone projekcje w `Params` → uczone `w_gen` w stanie (U.4).

**Zostaje (dobre priory, nie bugi):**
- Biologiczny prior makro-architektury (jakie regiony, role) — genetyczny
  blueprint. PC działa NA nim.
- Stałe `w_dn_motor` (wyjście filtra móżdżku, Dean-Porrill) — uzasadniony wyjątek.
- Sen/replay, oscylacje, ucieleśnienie (MJX), bufory opóźnień — wbudowane w FEP.
- Spike'owy `error_neuron.py` — zostaje jako wariant biofizyczny tam, gdzie
  potrzebny; rdzeniem inferencji staje się rate-mode `PCModule`.

---

## 5. Mapowanie na istniejący kod

| Moduł dziś | Rola w Fazie U | Akcja |
|------------|----------------|-------|
| **(nowy) `core/pc_module.py`** | kanoniczny `PCModule` (μ/ε/Π) z relaksacją | **utwórz (U.1)** |
| **(nowy) `core/pc_graph.py`** | `PCGraph` + pętla relaksacji | **utwórz (U.3)** |
| `core/error_neuron.py` | wzorzec pól; wariant spike'owy | zachowaj; nie-rdzeń |
| `core/cortex.py` | stos PC na laminach (Bastos 2012) | przepisz na `PCModule` (U.2) |
| `core/world_model.py` | węzeł grafu (generatywny PC) | wepnij jako pierwszy węzeł |
| `core/cerebellum.py` | przedni model PC; `w_gp` regułą wspólną | przepisz LTD→`Π·ε·μ` (U.2) |
| `core/precision_bus.py` | Π wektorowe w `PCModule` + kompozycja nagrody | rozszerz na wektor |
| `core/free_energy.py` | **konsument** — `variational_free_energy` + `graph_free_energy` + EFE | przywróć/wepnij (U.3/U.5) |
| `core/basal_ganglia.py` | precyzja polityk (Friston 2017) | przepisz aktor-krytyk (U.2/U.5) |
| `core/vta.py` | temporalny PC wartości; DA = Π | przepisz TD (U.2/§6) |
| `core/m1.py` | active inference: komenda = predykcja | usuń REINFORCE (U.5) |
| `core/neuromodulator.py` | bus precyzji (ACh/NE/DA/5-HT) | przepisz na kontrolery Π (U.5) |
| `core/hippocampus.py`, `sleep_replay.py` | szybkie PC + generatywny replay | przepisz one-shot (U.2) |
| `core/sparse.py` | wzrost/przycinanie grafu wg ΔFE + koszt | wepnij (martwy dziś) (U.4) |
| `core/brain_graph.py` | host grafu; relaksacja zamiast sekwencji | zastąp `cognitive_step` (U.3) |

Wniosek: kod zawiera prymitywy strukturalne PC (populacje μ/ε, precyzja,
generatywny world-model, kWTA), ale **brakuje rdzenia dynamicznego**
(relaksacja, globalna FE, graf). Faza U dobudowuje rdzeń i przepisuje reguły
na jego bazie — to scalenie pod jedną zasadą, nie kosmetyka.

---

## 6. Kredyt czasowy (integralna część, nie dodatek)

PC daje głęboki kredyt PRZESTRZENNY (w obrębie cyklu, przez relaksację). Dla
zadań rozciągniętych w czasie:
- **Dynamic / generalized predictive coding** — każdy węzeł ma DODATKOWĄ
  krawędź czasową `w_dyn` przewidującą własny następny stan (uogólnione
  współrzędne, Friston 2008). To realizuje cel TD (U.2 VTA) i predykcję
  sekwencji jako PC, nie osobny mechanizm. `world_model` już przewiduje
  następny stan — uogólniamy na cały graf.
- **Ślady kwalifikowalności** (idea z e-prop, Bellec 2020; Yagishita 2014 okno
  0.3–2 s) na wolnych krawędziach — most między zdarzeniem a opóźnioną
  nagrodą/błędem. Bufory `delay_*` (`brain_graph.py:155`) realizują opóźnienia
  przewodzenia; ślady dokładamy jako wolny stan krawędzi.

---

## 7. Dlaczego to skaluje (jawnie)

1. **Credit assignment:** PC ≈ backprop (Whittington & Bogacz 2017) **pod
   warunkiem relaksacji**, którą U.1 wprowadza i test U.1(b) weryfikuje →
   kredyt nie degraduje się z rozmiarem jak node-perturbation. Ściana #2.
2. **Struktura:** graf rośnie/przycina się sam (U.4) na dowolnej topologii
   (Salvatori 2022) → skalowanie = replikuj `PCModule` + pozwól strukturze
   wyłonić się. Ściana #1.
3. **Cel:** jedno równanie (FE/EFE) napędza percepcję, uczenie, akcję, uwagę i
   eksplorację → nowe zdolności = głębsza hierarchia, nie nowy kod.

To spójna (wciąż NIEUDOWODNIONA empirycznie na skali) ścieżka do neuro-AGI.

---

## 8. Ryzyka — przyjęte świadomie, mitygowane

Cel to AGI; ta faza to duża, jednoczesna przebudowa reguł uczenia. Ryzyka są
realne; przyjmujemy je i mitygujemy, nie udajemy, że ich nie ma.

1. **PC ≈ backprop tylko pod założeniami.** Na dużą skalę PC bywa wolniejszy/
   trudniejszy do strojenia (aktywny problem 2022–2025). *Mityg.:* test U.1(b)
   jako twarda bramka; Incremental + amortyzowana inicjalizacja μ.
2. **Stabilność relaksacji online.** Za mało kroków = zła inferencja; za dużo =
   wolno/niestabilnie. *Mityg.:* N=1–4 strojone; monitor zbieżności `F` (U.1a).
3. **Plastyczność strukturalna niestabilna.** *Mityg.:* twardy koszt okablowania
   + homeostaza + odłożenie U.4b na koniec.
4. **Active inference dla bogatej motoryki — otwarte.** Planowanie EFE rośnie
   wykładniczo bez przycinania. *Mityg.:* płytka hierarchia EFE na start;
   przycinanie polityk precyzją; reach jako pierwszy benchmark.
5. **Przepisanie wszystkich reguł naraz = ryzyko regresji wszystkiego.**
   *Mityg. (kluczowa):* migracja region-po-regionie z testem równoważności C1–C6
   na każdym kroku; `PCGraph` rośnie obok działającego `cognitive_step` aż go
   odtworzy. Nigdy nie usuwamy starej ścieżki przed zazielenieniem testu nowej.

---

## 9. Kryteria sukcesu — zdolności, nie hydraulika

Mierzone na rozszerzeniu `phase6b_capability.ipynb`:
1. **Relaksacja działa:** `F` modułu maleje w `pc_relax`, residual punktu stałego
   < 1e-3 (U.1a). ✅ ZREALIZOWANE.
2. **Kredyt = backprop:** cosinus gradientu PC (fixed-prediction) vs `jax.grad` =
   1.0000 (U.1b) — twarda bramka całej fazy. ✅ ZREALIZOWANE
   (`tests/test_phaseu_pc_module.py`). Relaksacja online: ≈0.9 (udokumentowane
   przybliżenie; ścisły backprop = tryb fixed-prediction).
3. **Unifikacja:** jeden `PCModule` + jedna reguła budują wszystkie regiony;
   liczba osobnych reguł uczenia = 1 (+ stałe `w_dn_motor`).
4. **Skalowanie kredytu:** głęboka hierarchia (≥4 poziomy) uczy się zadania,
   którego node-perturbation NIE uczył (np. predykcja wielokrokowa).
5. **Samo-okablowanie:** graf zmienia łączność i poprawia FE bez ręcznego
   dodawania krawędzi, przy stabilnej gęstości.
6. **Active inference:** reach BEZ node-perturbation; ciekawość = składnik
   epistemiczny EFE.
7. **Globalna FE maleje** na held-out doświadczeniu — jedna liczba, jeden cel.
8. **Brak regresji** C1–C6 po każdej migracji regionu.

---

## 10. Kolejność implementacji (migracja)

Buduj nowy rdzeń obok starego; migruj region po regionie; każdy krok ma test.

1. **U.1** ✅ — `core/pc_module.py`: relaksacja + Π + uczenie + tryb
   fixed-prediction. Bramka U.1(a)+(b) zielona (`tests/test_phaseu_pc_module.py`,
   4/4). Następny: krok 2.
2. **U.3 szkielet** ✅ — `core/pc_graph.py` (graf o dowolnej topologii, jedna
   reguła, relaksacja, `graph_free_energy`) + `free_energy.variational_free_energy`
   przywrócone. Test `tests/test_phaseu_pc_graph.py` 4/4: relaksacja redukuje
   globalną FE; jedna reguła uczy głębokiego łańcucha (§9.4); topologia cykliczna/
   wielorodzicielska relaksuje (§9.5).
3-5. **U.2 (big-bang)** ✅ substrat — `init_region_graph` instancjonuje WSZYSTKIE
   regiony (kora L1-L3, world_model, value/VTA, policy/BG, motor/M1, móżdżek, HC)
   jako węzły JEDNEGO grafu pod JEDNĄ regułą `Δw=η·Π·ε·φ(μ)`; pełny cykl
   clamp→relax→learn działa (test 4/4). **Pozostało:** (a) wpięcie grafu w żywą
   pętlę MJX (sensory/motor) = krok 8; (b) usunięcie legacy reguł spike'owych z
   `action_brain_cognitive_step` po dowodzie C1–C6. Legacy NIE usunięte — zostaje
   na dysku do czasu odtworzenia zdolności (§8.5).
6. **U.4a** ✅ — w grafie PC KAŻDA krawędź uczy się jedną regułą; zamrożonych
   projekcji nie ma. Spełnione z definicji substratu (krok 2). U.4b (strukturalna)
   = krok 9.
7. **U.5** ✅ — `core/pc_active.py`: action-as-inference (predictions not
   commands, Adams 2013) + `expected_free_energy` przywrócone z konsumentem
   (`pc_efe`/`efe_select`, argmin EFE = polityka BG) + neuromod jako kontrolery
   precyzji (`scale_node_precision`). `pc_brain_act`/`pc_brain_learn_forward`
   wpięte (krawędź motor→sensory). Test `tests/test_phaseu_pc_active.py` 4/4:
   **babble→reach BEZ REINFORCE** (forward model jedną regułą, reach błąd wzgl.
   <10%); predictions-not-commands; argmin EFE (greedy vs ciekawość); komenda
   mózgu zależna od celu.
8. **U.3 domknięcie** ✅ substrat — `core/pc_brain.py`: `pc_brain_cognitive_step`
   = cykl poznawczy jako relaksacja grafu (clamp sensory → relax → odczyt
   motoryki → jedna reguła), zamiast ręcznej sekwencji. Drop-in API zgodne z
   driverem MJX (płaskie `sensory` in, `tanh` joint command out). Test
   `tests/test_phaseu_pc_brain.py` 4/4: krok ograniczony/skończony; powtórna
   ekspozycja obniża globalną FE (percepcja się uczy); `learn=False` = czysta
   inferencja; odczyt motoryki zależy od sensory. **Pozostało (bramka C1–C6):**
   podmiana wywołania w `embodiment/mjx_run_loop.py` + usunięcie spike'owego
   pipeline'u — dopiero po odtworzeniu reach (§8.5).
9. **U.4b** ✅ — `core/pc_structural.py`: self-wiring wg FE — wzrost gdzie
   `|∂F/∂W|` duże (ta sama reguła), przycinanie `|W|<próg` (koszt okablowania),
   twardy budżet gęstości per krawędź (homeostaza, top-k wg priorytetu).
   Test `tests/test_phaseu_pc_structural.py` 3/3: self-wiring bije frozen-sparse
   (7 synaps loss 0.18 → 37 synaps loss 0.0); przycinanie usuwa słabe; cap gęstości
   trzyma. (`sparse.py` zostawione — niezależny scaffolding synapsy-poziom; graf
   ma własną strukturalną na poziomie maski krawędzi.)

**Czyszczenie legacy (wykonane):** usunięto/przywrócono `expected_free_energy`
zgodnie z zasadą „brak prymitywu bez konsumenta" (martwe po U.3 → wróciło z U.5).
Spike'owy per-region pipeline NIE usunięty: decyzja użytkownika + §8.5 (repo działa).

---

## 12. Decyzja: bramka równoważności vs rewrite (po U.1–U.5)

**Decyzja: ANI bit-parity bramka C1–C6 przeciw mózgowi spike'owemu, ANI rewrite
od zera. Zamiast tego: capability-acceptance grafu PC na realnym zadaniu
ucieleśnionym.**

Uzasadnienie:
1. **Bit-parity przeciw spike to zły cel.** Inny substrat (rate PC vs spike AdEx);
   odtwarzanie dynamiki spike'owej nie jest celem. §9 już definiuje sukces przez
   ZDOLNOŚCI (reach success, spadek FE, skalowanie kredytu), nie parytet hydrauliki.
2. **Rewrite od zera to marnotrawstwo.** Embodiment (ciało MJX, kodowanie
   sensory/proprio) jest substrate-agnostic i wielokrotnego użytku; zmienia się
   tylko wywołanie mózgu.
3. **Ścieżka: zostaw embodiment, podmień mózg.** Adapter MJX napędza `pc_brain`
   (sensory→clamp, `pc_brain_act`→komenda, `pc_brain_learn_forward` z realnego
   proprio). Walidacja na **reach success** (realna zdolność, nie parytet). Gdy
   `pc_brain` przejdzie próg zdolności → usuń `brain_graph` + moduły per-region +
   ich testy phase.

To NIE jest bramka równoważności — to **próg zdolności**. Następny duży krok
(osobny, poza Fazą U-core): adapter `embodiment/` dla `pc_brain` + eksperyment
reach w MJX. Faza U-core (U.1–U.5, substrat) jest kompletna i przetestowana
(19/19).

---

## 11. Sedno

Obecny system to dobry biologiczny PROTOTYP z właściwą filozofią, ale bez
rdzenia dynamicznego PC i z sześcioma osobnymi regułami — stąd plateau. Faza U
**dobudowuje prawdziwą relaksację swobodnej energii** (U.1), **przepisuje
wszystkie reguły na jedną** na **rosnącym grafie** (U.2–U.4) i czyni **akcję
active inference** (U.5). To nie reinterpretacja — to wymiana mechanizmu
uczenia i przepływu na jeden substrat. Ryzyko jest duże i przyjęte świadomie:
celem jest AGI, a to jedyna z rozważanych zmian, która rusza WIĄŻĄCE
ograniczenie (siła i skalowalność uczenia), nie tylko sztywność struktury.
Twardą bramką jest test U.1(b): jeśli PC nie da kredytu na poziomie backpropu,
zatrzymujemy się i naprawiamy rdzeń, zanim pójdziemy dalej.
