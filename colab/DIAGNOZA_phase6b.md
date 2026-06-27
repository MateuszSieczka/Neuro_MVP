# Diagnoza Phase 6B — pipeline, oczekiwane wyniki, błędy logiczne, ocena testów

> Dokument analityczny. **Nie zmienia logiki mózgu** (`core/`, `embodiment/`).
> Opisuje co robi obecny kod, co testy *powinny* dawać, jakie błędy w logice
> mogą uniemożliwić poprawny wynik, oraz czy testy w ogóle mierzą ten typ
> inteligencji, który chcesz zbudować.

---

## 1. Pipeline — przepływ jednego cyklu poznawczego

Wejście: `action_brain_cognitive_step(state, params, ctx, sensory, prev_reward, prev_done, key)`.
Sterownik ciała: `embodiment/mjx_run_loop.py` (`_one_babble_cycle` / `_one_reach_cycle`).

Kolejność operacji w jednym cyklu (`core/brain_graph.py:1200`):

1. **Sen / czuwanie** (`0a`): integrator adenozyny (Process-S) → `sleep_step`
   ustala fazę WAKE/SWS/REM. `energy = 1 − adenosine`.
2. **Reset PFC na granicy epizodu** (`0b`, gdy `done`).
3. **Percepcja** (`1`, `jax.lax.scan` po `substeps`, domyślnie 20):
   oscylator → talamus(`sensory`) → kora (L4→L2/3→L5) → móżdżek → krytyk →
   aktory BG → predykcja world-model → neuromodulatory. Zwraca
   `cortex_belief`, `cortex_l5_rate`, `relay_spikes`.
   Kora dostaje `ff_input = thal.relay_spikes` (czyli `sensory`), **bez** wejścia
   top-down. Dla reachera `sensory` = proprio (kąty+prędkości, kod populacyjny)
   ⊕ kod populacyjny wektora `target − tip`.
4. **Pamięć hipokampalna** (`1b`): EC → DG→CA3→CA1. Mismatch CA1 → bonus ACh.
5. **World-model update** (`2a`): uczenie na `(s_t, a_t, s_{t+1})`; błąd predykcji
   sensorycznej `sensory_pe → climbing_sensory` (włókna pnące, projekcja
   `w_io_pc`).
6. **Proprio + motor PE** (`2a.2`): `motor_pe = enc(qpos_norm_actual) −
   enc(predicted)`, gdzie `predicted = jc` z poprzedniego cyklu →
   `climbing_motor` (`w_motor_pc`). `cerebellum_update` uczy `w_gp`.
7. **Ciekawość** (`2b`): `curiosity = wm.pe_short_abs`. **`r_total = r_ext`**
   (ciekawość świadomie NIE wchodzi do V(s) — by nie skażać krytyka, Pathak 2017).
8. **VTA RPE** (`2c`): TD(0) `rpe = (r − baseline) + γV(s') − V(s)`, normalizacja
   D2 (auto_rms).
9. **Uczenie wag** (`2d`): VTA, krytyk, aktory (aktor ciała dostaje bonus
   `+curiosity_z`, aktor sakkad `+info_gain_z`), STDP kory. Replay store (`2g`).
10. **Wybór akcji BG** (`4`).
11. **M1** (`4a`): `cb_correction = dn_rate @ w_dn_motor`;
    `jc = tanh(l5·W + α·tanh(cb_correction) + ξ)`, `ξ ~ 𝒩(0, σ(NE))`.
    `m1_learn_readout`: node-perturbation REINFORCE
    `Δw = lr·(rpe − baseline_local)·outer(l5, ξ)`. `predicted_angles_next = jc`.

Ciało (`mjx_arm_body.act_continuous`): `ctrl = jc · joint_range`, serwo
pozycyjne (`kp=20, kv=4`), `frame_skip=3` kroki po 10 ms = 30 ms/cykl.
Potem `_sync_brain_to_body` zapisuje `qpos/joint_range → last_joint_angles`
i `jc → m1.last_joint_command`.

Nagroda w reach: `r_shaped = (prev_dist − dist) + (1.0 jeśli dist<0.05)`,
podana jako `prev_reward` w następnym cyklu.

Jednostki są spójne po fixach D9: `predicted_angles_next = jc ∈ [−1,1]` oraz
`last_joint_angles = qpos/joint_range ∈ [−1,1]` — oba znormalizowane, więc
motor PE liczony jest w tych samych jednostkach (to był sens „Phase 1 fix").

---

## 2. Co testy POWINNY dawać (kryteria z `new_plan.md`)

| Test | Realne kryterium planu | Oczekiwany wynik (po naprawach §3) |
|------|------------------------|------------------|
| `test_phase6b_mjx_jit_speed` | 1000 cykli < 30 s (Win: 60 s) po warm-up | przejście — to test wydajności |
| `test_phase6b_babbling_coverage` | ≥40% komórek 2D workspace po babbling | **realne** — szum ξ M1 + szeroki `ctrlrange ±2.0` pokrywają pierścień |
| `test_phase6b_cerebellum_forward_mse` | motor PE ≤ 40% wartości początkowej | **osiągalne** — pętla modelu przedniego zamknięta (§3.1/3.2); `w_gp` redukuje błąd wyjścia |
| `test_phase6b_reach_success` | success rate (dist<0.05 w 500 krok.) ≥ 0.4 przy random ≤ 0.05, 2000 epizodów | **prawdopodobne** — poprawny score polityki (§3.3) + działający model przedni + babbling skupiony na modelu przednim (§3.4); wciąż wymaga eksperymentu 2000-epizodowego w Colab |
| `test_phase6b_sleep_cycle` | success po śnie ≥ 1.15× przed snem | **nieweryfikowane przez pytest** — patrz §4 |

Diagnostyczny `verify_phases.ipynb` (już uruchomiony) pokazuje, że babbling
post-fix jest zdrowy: `jc` nie jest nasycone (0.17–0.88, zróżnicowane),
`pred_q == jc` (diff 0), `dn_rate ≈ 5`, `V` rośnie 0.01→0.30, `auto_rms`
spada 0.98→0.69, `|W|` stabilne ≈ 0.35. To znaczy, że **wcześniejsze błędy
(nasycenie tanh, błędne stałe czasowe EMA) zostały naprawione**.

---

## 3. Błędy / luki logiczne — ZNALEZIONE I NAPRAWIONE

Nie było pojedynczego „śmiertelnego" buga czyniącego reach *niemożliwym*; były
**strukturalne luki**, które ograniczały zdolność sięgania i sprawiały, że nawet
zielone testy nie dowodziły uczenia. Wszystkie pięć naprawiono profesjonalnie,
wg papierów, bez magicznych stałych i bez kodu „mostkującego". Status każdej:
**NAPRAWIONE**.

### 3.1 + 3.2 Móżdżek = poprawny model przedni (adaptacyjny filtr), pętla zamknięta

**Było:** `w_dn_motor` (jądra głębokie→komenda), `w_motor_pc` (PE→Purkinje) i
`w_io_pc` (PE sensoryczne→Purkinje) były **stałymi losowymi** projekcjami;
uczyło się tylko `w_gp`. `w_dn_motor` było dodatkowo jednoznakowe (`|𝒩|`), więc
korekta mogła pchać tylko w jedną stronę. `predicted_angles_next = jc` ignorowało
móżdżek — pętla modelu przedniego była otwarta. Wynik: kierunek korekty losowy,
móżdżek nie był modelem Wolperta, motor-PE zdominowany przez stały bias serwa.

**Naprawione** (wg **adaptacyjnego filtra móżdżku** — Dean, Porrill, Ekerot &
Jörntell 2010, *Nat Rev Neurosci* 11:30; Porrill, Dean & Stone 2004; Fujita
1982; Wolpert 1998):

- Móżdżek jest teraz **modelem przednim motoryki**; predykcję sensoryczną
  oddano world-modelowi (czysty podział pracy, Wolpert 1998). Usunięto
  `climbing_sensory`, `w_io_pc`, `w_motor_pc`, `w_io_pc_sigma`, `m1_cb_alpha`
  (martwy kod).
- `w_dn_motor` to **stała** macierz wyjściowa filtra adaptacyjnego (w tym modelu
  wagi wyjściowe są stałe; plastyczność jest na synapsie włókno równoległe→
  Purkinje, czyli `w_gp`). Teraz **znakowana** (dwukierunkowa) i skalowana
  fan-in `1/√n_dn` (LeCun 1998) — bez magicznej stałej.
- Błąd uczący (włókna pnące) liczony w **przestrzeni stawów** i sprzężony
  transpozycją: `climbing = (q_actual − q_predicted) @ w_dn_motor.T`. To
  warunek konieczny i wystarczający, by kowariancyjna reguła LTD w
  `cerebellum_update` redukowała błąd **wyjścia** (Porrill, Dean & Stone 2004
  — sterowanie dekorelacyjne: nauczyciel musi docierać do każdej komórki
  Purkinjego w jej własnym układzie współrzędnych wyjścia).
- **Pętla zamknięta:** `predicted_angles_next = clip(jc + cb_forward, −1, 1)`,
  gdzie `cb_forward = dn_rate @ w_dn_motor`. Móżdżek uczy się odchylenia od
  idealnego śledzenia serwa (transient opóźnienia) — kanoniczny sygnał błędu
  modelu przedniego Wolperta (1998). Gdy `w_gp` się uczy, motor-PE → 0, więc
  realne kryterium `cerebellum_forward_mse` staje się osiągalne.
- M1 **nie** dostaje już korekty móżdżkowej: wyjście modelu przedniego ma zły
  znak jako korekta komendy (przewidywany niedoskok *zmniejszyłby* komendę) i
  myliłoby model przedni z odwrotnym. Mapę odwrotną (stan→komenda) uczy reguła
  REINFORCE odczytu M1.

Pliki: `core/brain_graph.py` (init + blok motor-PE + blok M1), `core/m1.py`.

### 3.3 M1 REINFORCE = poprawny score Gaussa (Williams 1992)

**Było:** `Δw = lr·(rpe−b)·l5·ξ`. Brakowało dzielenia przez wariancję eksploracji.
Ponieważ σ jest bramkowane przez noradrenalinę, ten sam ξ dawał gradient o
złej skali przy zmianie pobudzenia → zawyżona wariancja i bias w stronę epizodów
o wysokim NE. To nie był poprawny score polityki Gaussa.

**Naprawione:** dla polityki `a = μ + ξ`, `ξ ~ 𝒩(0, σ²)`, `μ = wᵀl5`, score to
`∂lnπ/∂w = l5·ξ/σ²` (Williams 1992, wzór dla rozkładu normalnego; Fiete & Seung
2006 node perturbation). Reguła to teraz `Δw = lr·(rpe−b)·(σ_base/σ)²·l5·ξ`, z
wariancją odniesienia `σ_base²` wchłoniętą w `lr` (skala stabilna przy
σ = σ_base, poprawne re-ważenie próbek przy innym σ). σ ma dodatnią podłogę
(`sigma_floor > 0`, baza LC; Aston-Jones & Cohen 2005), więc `1/σ²` jest zawsze
skończone. σ jest teraz cache'owane na stanie M1 (`last_sigma`).

Pliki: `core/m1.py` (`m1_step`, `m1_learn_readout`, `M1State`).

### 3.4 Brak nagrodowego uczenia M1 w babblingu (separacja rozwojowa)

**Było:** babbling karmił `prev_reward = curiosity` i pozostawiał `readout_lr`
aktywne, więc REINFORCE M1 uczył się **maksymalizować zaskoczenie**, nie sięgać.
W sygnaturze tkwiły martwe kwargs `ou_tau`/`ou_sigma`.

**Naprawione** (wg Oller 1980 canonical babbling; von Hofsten 2004; plan, decyzja
#6): babbling to self-supervised akwizycja modelu przedniego, *poprzedzająca*
sterowanie celowe. Dodano bramkę `learn_motor_readout: bool = True` w
`action_brain_cognitive_step`; pętla babblingu ustawia ją na `False` i podaje
`prev_reward = 0`. Eksploracja pochodzi z szumu ξ M1 (zmienność kory ruchowej;
Tumer & Brainard 2007). Móżdżek i world-model uczą się dalej (ich plastyczność
nie jest bramkowana nagrodą). Usunięto martwe `ou_tau`/`ou_sigma`; carry
babblingu nie nosi już nagrody.

Pliki: `core/brain_graph.py`, `embodiment/mjx_run_loop.py`
(`_one_babble_cycle`, `_babble_chunk`, `run_babbling`).

### 3.5 Ciekawość jako nagroda = learning progress (Oudeyer 2007), nie surowe |PE|

**Było:** bonus aktora ciała = `wm_curiosity_signal` = `pe_short_abs` (surowe
|PE|). Surowe |PE| nagradza też błąd nieredukowalny („noisy-TV") i kolapsuje, gdy
świat staje się przewidywalny.

**Naprawione** (wg Oudeyer & Kaplan 2007 IAC; Schmidhuber 1991): bonus aktora to
teraz **learning progress** `pe_long − pe_short` (już zaimplementowane w
`wm_learning_progress`), z-skorowane przez kanał precyzji. Nagradza front uczenia,
jest odporne na noisy-TV (region o nieredukowalnym |PE| ma ≈0 LP → brak bonusu),
a zanika tylko tam, gdzie nie ma już czego się uczyć (zamierzona „nuda" pchająca
agenta dalej). Surowe `pe_short` zostaje jako napęd zaskoczenia dla ACh/NE
(Yu & Dayan 2005 — odrębny sygnał) i jako salience replay.

Pliki: `core/brain_graph.py` (kompozycja nagrody aktora).

---

## 4. Czy testy testują ten typ inteligencji, który chcesz zbudować?

**Krótko: nie. Testy weryfikują, że pipeline ŻYJE i jest liczbowo zdrowy —
nie że pojawia się zamierzona inteligencja.**

Każdy „semantyczny" test Phase 6B został w pytest zdegradowany do taniego proxy:

- `reach_success` → asercja `brain_mean ≤ rand_mean × 1.05`
  (`tests/test_phase6b_reach_success.py:69`). To znaczy: „mózg nie jest gorszy
  od losowego o >5%". **Mózg który NIC nie robi** (trzyma stałą pozę blisko
  środka) łatwo to przejdzie, bo polityka losowa szarpie szeroko → duża średnia
  odległość. Test **nie mierzy uczenia sięgania**. Realne kryterium (success ≥ 0.4)
  jest celowo przeniesione do notebooka Colab.
- `cerebellum_forward_mse` → asercja `‖Δw_gp‖ > 1e-3` + `dn_rate` skończone.
  Każde niezerowe `lr` rusza wagi. Test **nie mierzy spadku MSE** modelu przedniego.
- `sleep_cycle_improves_reach` → asercja `|Δatp| > 1e-6` + brak NaN. **W ogóle nie
  mierzy poprawy po śnie.**
- `babbling_coverage` → to jedyny test mierzący realne kryterium (≥40% pokrycia).

Wniosek: **zielony zestaw pytest ≠ „AI działa".** Prawdziwe kryteria zdolności
istnieją wyłącznie w notebookach Colab, które muszą być **uruchomione i
przeczytane przez człowieka**. To jest największa luka diagnostyczna.

Czego żaden test (pytest ani notebook) NIE mierzy, a co definiuje Twój cel:

1. **Rozumienie świata** — dokładność predykcji world-model na *trzymanych*
   (held-out) przejściach, nie tylko `pe_short` na trajektorii treningowej.
2. **Napęd ciekawości** — czy agent *preferencyjnie* odwiedza regiony o wysokim
   learning-progress (`wm_learning_progress`), czy tylko ma niezerowy sygnał.
3. **Uczenie z interakcji** — krzywa uczenia reach (success rate vs epizod)
   z istotnym marginesem nad losowym, oraz ablacja `readout_lr=0` dowodząca,
   że to *uczenie* jest przyczyną poprawy, a nie geometria/inicjalizacja.
4. **Transfer / generalizacja** — np. forward-model wytrenowany w babblingu
   redukuje błąd na *nowych* celach.

Architektura jest natomiast **zgodna z Twoją ambicją**: lokalne reguły uczenia
(STDP, three-factor, node-perturbation REINFORCE, Marr-Albus LTD), neuromodulatory,
sen/replay, ciekawość typu learning-progress (Oudeyer/Schmidhuber), rama active
inference, brak backpropu. Problem nie leży w filozofii kodu — leży w tym, że
**testy mierzą hydraulikę, nie zdolności poznawcze**.

---

## 5. Wpływ napraw na notebooki diagnostyczne

Naprawy zmieniają semantykę, którą weryfikował `phase6b_verify_phases.ipynb`:

- **V1 / V4 są nieaktualne.** Weryfikowały „P1: `predicted_angles_next = jc_out`"
  oraz „`pred_q == jc`". Po §3.2 predykcja to `jc + cb_forward`, więc te asercje
  z założenia będą fałszywe — to zamierzone (pętla modelu przedniego jest teraz
  zamknięta). Te komórki należy usunąć/zastąpić pomiarem spadku motor-PE.
- **V3 wymaga drobnej zmiany sygnatury:** `_babble_chunk` nie przyjmuje już
  `prev_r` (babbling nie nosi nagrody); carry to `(brain_state, body, sensory,
  prev_d)`. Wywołanie w notebooku trzeba zaktualizować.
- `phase6b_diag.ipynb` używa własnego `diag_babble_scan` (woła
  `action_brain_cognitive_step` bezpośrednio) — działa nadal; jego D3.3 woła
  `run_babbling` bez `ou_*`, więc też działa.

## 6. Rekomendacje (dalsze, diagnostyczne)

1. **Dodać notebook „capability" w Colab**, który mierzy realne kryteria:
   krzywa uczenia reach (success rate / meanD vs epizod) z baseline losowym
   i oracle; ablacja `learn_motor_readout=False` vs aktywne (dowód, że uczenie
   readoutu pomaga); forward-model MSE na held-out seed (realne kryterium 6B);
   test kierunku ciekawości (korelacja czas-w-regionie z learning-progress).
2. **Sondowanie L5**: sprawdzić, czy `cortex_l5_rate` różnicuje stany po pozycji
   celu (warunek konieczny, by M1 mógł sięgać) — eff-rank i korelacja L5 z
   `target − tip`. Diag F8 dotyka tego (`l5_top1`), ale nie wprost względem celu.
3. **Wzmocnić pytesty** o realne kryteria (small-scale): reach > random o istotny
   margines (nie tylko ≤1.05×), spadek forward-MSE, poprawa po śnie — by zielony
   pytest faktycznie świadczył o zdolności, nie tylko o hydraulice (§4).

Notebook z punktu 1 mogę dla Ciebie napisać — powiedz słowo.
