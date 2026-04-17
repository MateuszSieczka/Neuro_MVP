# Plan: Phase 3 + Phase 4 — od MinimalBrain do aktywnego, uczącego się embodied agenta

## TL;DR

Dwie kolejne fazy mają zamienić obecny `MinimalBrain` (sensory→thalamus→cortex→cerebellum, czysto pasywny) w aktywnego agenta z zamkniętą pętlą **sense → act → learn → reward** oraz **skalowalną, aktywną percepcją**.

- **Phase 3 — Embodied Action Loop + Intrinsic Drives.** Zamykamy najprostszą możliwą pętlę: BG+VTA+body_interface, inferior-olive proxy dla cerebellum, wewnętrzna motywacja (curiosity/boredom/homeostasis). Weryfikacja na trywialnym `GridWorld`/`ContinuousBanditWorld`.
- **Phase 4 — Scalable Active Perception (vision + audio szkielet).** Foveal + multi-resolution retinal pyramid (scale-invariant), uczona V1 sparse coding zamiast fixed Gabor, saccade jako mental action wybierana przez BG z Phase 3, wspólny szkielet dla cochlea+auditory attention. Babbling i naming świadomie przesunięte do Phase 5.

Obie fazy są niezbędne ale niewystarczające do AGI — cel: zdolność do **autonomicznego zbierania doświadczeń** (Phase 3) i **przeszukiwania przestrzeni sensorycznej** (Phase 4), na których opiera się cała dalsza emergencja konceptów.

---

## Phase 3 — Embodied Action Loop + Intrinsic Drives

### Cel

Agent musi móc **wybrać akcję**, **wykonać ją** na ciele, **odebrać skutek** i **uczyć się z niego**. Bez zamkniętej pętli nic więcej (koncepty, język, nawigacja) nie ma sensu uczeniowego. Zaczynamy od najprostszego środowiska bez fizyki — dopiero po zamknięciu pętli rozważamy MuJoCo/Unity.

### Zakres (co WCHODZI)

1. **`core/body_interface.py`** — ABC + stub adapter
   - Klasa `BodyInterface` z metodami: `sense() -> SensorySample`, `act(motor_command) -> None`, `reset() -> SensorySample`.
   - `SensorySample` = NamedTuple (sensory: jnp.ndarray, reward: float, done: bool, info: dict).
   - Adapter #1: `GridWorldBody` — prosty 5×5 grid z jedną nagrodą, 4 akcje dyskretne, sensory = one-hot pozycji + relatywny wektor do celu.
   - Adapter #2: `ContinuousBanditBody` — k ramion z gaussowskimi reward'ami, sensory = kontekst + last_reward, akcja = indeks ramienia. Najprostszy test VTA.
   - `MuJoCoBody` / `UnityBody` — **NIE** w tej fazie, tylko stub z `NotImplementedError` i komentarzem roadmap.

2. **Integracja BG + VTA w `brain_graph.py`** — nowy `ActionBrain` (rozszerzenie `MinimalBrain`)
   - Bierze istniejące `core/basal_ganglia.py` (actor/critic D1/D2) i `core/vta.py` (RPE → dopamina).
   - Wiring: cortex L5 rate → striatum_input (actor); cortex L2/3 belief → VTA critic state; VTA dopamine → neuromodulator.da bus (tonic + phasic).
   - BG wybiera **dyskretną akcję** (one-hot n_actions) z rozkładu softmax wartości D1-D2.
   - Dla ciągłych akcji (Phase 4+): dodać dodatkowy "continuous head" — pominięte teraz.
   - Nowy bufor opóźnienia: BG→M1 ≈ 5 ms (cortico-striato-thalamo-cortical loop, Haber & Calzavara 2009).

3. **Motor output path przez thalamus VA/VL**
   - Plan.md planował M1 osobno. Na tym etapie: `action = bg.actor_choice` wychodzi bezpośrednio z `brain_graph` przez pole `MinimalBrainOutput.motor_command`.
   - Pełny M1 jako `CorticalArea` — Phase 5. Teraz wystarczy dyskretny one-hot.
   - Efference copy: kopia `motor_command` wraca do `cerebellum.mossy` jako efference (uzupełnienie do proprioception).

4. **Inferior olive proxy (cerebellum training signal)**
   - Aktualnie `climbing_error` jest external. Teraz liczymy ją endogenicznie:
     `climbing_error = sensory_t − predicted_sensory_{t-1}`
     gdzie `predicted_sensory` to wyjście cerebellum z poprzedniego kroku (mapping granule → predicted_sensory przez drugą wagę readout, obok `w_gp`).
   - Uwaga: to WYMAGA rozszerzenia `CerebellumParams` o `w_sensory_predict` (readout z granule → sensory_size).
   - Alternatywa (prostsza): korzystać z istniejącego `core/world_model.py` jako proxy IO — jego PE = climbing signal. Wybór: **druga opcja**, bo `world_model.py` już to robi prawidłowo, unikamy duplikacji.

5. **Intrinsic drives** (reward shaping wewnętrzny)
   - **Curiosity** = średnie PE z `cortex.l23_error` (już mamy, wystarczy routing do VTA jako dodatkowy bonus reward).
   - **Boredom / information gain drop** = spadek PE w czasie (moving average delta) → sygnał "ucz się czegoś innego", wpływa na NE.
   - **Homeostasis** = astrocyta ATP poniżej progu → negatywny reward (dryf w stronę zachowań energetycznie tanich). Phase 3 tylko PODŁĄCZA astrocyta do reward bus (nie implementuje jeszcze snu — to Phase 5).
   - Całkowity reward dla VTA: `r_extrinsic + β_curiosity·PE + β_homeo·atp_deviation`. Współczynniki: `β_curiosity=0.1`, `β_homeo=0.05` (krytycznie małe; zewnętrzny reward dominuje — Barto et al. 2013 argumentowali, że curiosity to drugorzędny sygnał gdy reward jest dostępny).

6. **`arena/` → `embodiment/`** — mały refactor
   - Usuwamy stary `arena/snn_agent.py`, `agent_factory.py`, `benchmark.py`, `gym_env.py` (są RL-centric i kolidują z nowym podejściem).
   - Tworzymy `embodiment/` z `body_interface.py`, `gridworld.py`, `bandit.py`, `run_loop.py` (jeden step loop wywołujący `minimal_brain_step` + `body.act`).

7. **Testy integracyjne**
   - `tests/test_phase3_bandit.py` — `ContinuousBanditBody` z 3 ramionami, expected: po ≥500 krokach agent wybiera best arm z prob ≥0.7.
   - `tests/test_phase3_gridworld.py` — 5×5, start w rogu, cel w przeciwnym. Expected: po ≥2000 krokach średnia długość epizodu spada o ≥30% vs random policy.
   - `tests/test_phase3_curiosity.py` — agent w środowisku BEZ reward. Expected: non-zero exploration, cortex PE nie zanika do zera (curiosity drive działa).
   - `tests/test_phase3_olive_proxy.py` — cerebellum uczy się predykcji sensory po ≥1000 krokach (MSE spada ≥40%).

### Zakres (co NIE wchodzi)

- Żadnej fizyki (MuJoCo/Unity) — świadomie. Najpierw correctness, potem rendering.
- Żadnego M1/premotor jako `CorticalArea` — Phase 5.
- Żadnej wizji/audio — Phase 4.
- Żadnego snu / replay — Phase 5.
- Żadnej strukturalnej plastyczności — odrzucone do Phase 6+ aż topologia fixed udowodni stabilność.

### Krytyczne pliki do stworzenia/modyfikacji

- `core/brain_graph.py` — dodać `ActionBrainParams/State/Output`, `init_action_brain_params`, `action_brain_step` (rozszerzenie `minimal_brain_step` o BG+VTA+world_model).
- `embodiment/__init__.py` — nowy pakiet.
- `embodiment/body_interface.py` — ABC + `SensorySample`.
- `embodiment/gridworld.py`, `embodiment/bandit.py` — dwa minimalne adaptery.
- `embodiment/run_loop.py` — `run_episode(brain_state, body, steps)` → trajectory.
- `core/basal_ganglia.py` — mały lift: upewnić się że API jest `bg_step(state, params, ctx, striatum_input, da_phasic)` zgodne z konwencją; dodać `bg_reset_transient`.
- `core/vta.py` — lift API do tego samego kształtu; dodać rozdzielenie tonic vs phasic DA do neuromodulator bus.
- `core/world_model.py` — udostępnić `prediction_error` jako climbing signal dla cerebellum (lub refactor: cerebellum sam konsumuje world_model PE).
- Usunąć: `arena/snn_agent.py`, `arena/agent_factory.py`, `arena/benchmark.py`, `arena/gym_env.py`, `arena/task_config.py`, `arena/environments.py`. Zostawić tylko `arena/core.py` (ABCs) **jeśli** nie koliduje; w przeciwnym wypadku usunąć cały `arena/`.

### Kluczowe decyzje projektowe (Phase 3)

- **Dyskretne akcje first** — BG actor zwraca one-hot z n_actions. Ciągłe (drgające) akcje z M1 w Phase 5.
- **Brak osobnego M1 teraz** — cortex L5 + BG actor jest wystarczające dla GridWorld. Dodawanie M1 przed zamknięciem pętli jest over-engineeringiem.
- **World model jako inferior olive proxy** — wybór między duplikowaniem logiki w cerebellum a reużyciem `world_model.py`. Wybieramy reużycie; cerebellum dostaje `climbing_error = world_model.prediction_error`.
- **Intrinsic reward jest aditywny, nie zastępczy** — zewnętrzny reward zawsze dominuje. Curiosity tylko gdy reward=0 (w praktyce β·PE ≪ |r_extrinsic|).
- **Żadnego doświadczenia przechowywanego poza replay_buffer** — ale replay NIE jest używany jeszcze w Phase 3 (brak snu). Eligibility trace cortex/BG wystarcza dla on-line learning.

### Weryfikacja Phase 3

1. **Unit**: `test_action_brain_step_jit` — `action_brain_step` JIT-uje się i wykonuje ≥500 kroków bez OOM/NaN.
2. **Funkcjonalny**: bandit test (wybór best arm).
3. **Funkcjonalny**: gridworld test (redukcja długości epizodu).
4. **Funkcjonalny**: curiosity test (exploration bez reward).
5. **Funkcjonalny**: olive proxy test (cerebellum uczy się sensory prediction).
6. **Diagnostyczny**: spike-rate hygiene — cortex 2-10 Hz, BG MSN 1-5 Hz, VTA 3-8 Hz; brak runaway.
7. **Diagnostyczny**: dopamina bus — phasic spike przy unexpected reward ≥3× tonic baseline.

### Szacowanie złożoności

Pliki nowe: ~5 (embodiment + testy). Pliki modyfikowane: ~4 (brain_graph, basal_ganglia, vta, world_model). Najtrudniejszy element: **JIT-owanie pełnej pętli z BG+VTA** (te moduły powstały w NumPy stylu; mogą wymagać odświeżenia do `eqx.Module`).

---

## Phase 4 — Scalable Active Perception (vision + audio szkielet)

### Cel

Rozwiązać problem sztywnej rozdzielczości 128×128, który wskazałeś. Percepcja ma być:

1. **Scale-invariant** — ten sam mózg przetwarza 64×64 i 4K bez zmiany topologii.
2. **Aktywna** — agent sam wybiera **gdzie** patrzeć (saccade) i **na co** słuchać (auditory attention window). To wymaga Phase 3 (bo "gdzie patrzeć" to akcja wybierana przez BG).
3. **Uczona bottom-up** — V1 to sparse coding uczone STDP (Olshausen & Field 1996), nie fixed Gabor. Gabor jest co najwyżej inicializacją.

### Zakres (co WCHODZI)

1. **`sensory/retina.py`** — scale-invariant retinal preprocessing
   - **Gaussian pyramid** z N poziomów (konfigurowalnymi) — input dowolnego rozmiaru, output stała liczba peryferyjnych "kafli" niskiej rozdzielczości.
   - **Foveal patch** — wycinek wysokiej rozdzielczości wokół `fixation_point` (parametr wejściowy, kontrolowany przez BG w Phase 4.3).
   - **DoG filters** (ON-center/OFF-center) na każdym poziomie piramidy i na fovea patch.
   - **Temporal differencing** (ramka t − ramka t−1) — event-like sparse spikes.
   - Output: `RetinalSample(fovea_spikes, pyramid_spikes, fixation_xy)` — **fixed-shape** niezależnie od input_size. To jest klucz do skalowalności.
   - Żadnego uczenia — retina jest fixed (biologicznie też: ganglion cells mają fixed receptive fields).
   - Reference: Rodieck (1998), Rosenholtz (2016) "Capabilities and limitations of peripheral vision", Itti & Koch (2001) saliency.

2. **`sensory/lgn_adapter.py`** — most do istniejącego thalamusu
   - `RetinalSample` → input dla `RelayParams` już istniejącej `core/thalamus.py`.
   - Konkat fovea + pyramid → płaski wektor `n_afferent` ≈ 256-512.
   - Burst/tonic już działa (ACh/NE z Phase 2).
   - Attentional gating: TRN modulator z PFC (Phase 5 doda PFC; teraz stały baseline).

3. **`sensory/v1.py`** — uczona warstwa sparse coding
   - `CorticalArea` z `core/cortex.py` jako `n_l4=256, n_l23_state=256, n_l23_error=64, n_l5=64`.
   - **Inicjalizacja wagami Gabor-like** (8 orientacji × 4 SF × ON/OFF = 64 filters, replicated z szumem) — startowy prior.
   - Uczenie przez istniejący STDP w cortex.py — PO iluzji tygodni ekspozycji receptive fields powinny się wyostrzyć w kierunku statystyk danych (Olshausen & Field 1996). W testach: nieformalnie po ≥10k kroków.
   - Żadnego backprop.
   - Reference: Olshausen & Field (1996), Hunsberger & Eliasmith (2015) dla spiking sparse coding.

4. **`sensory/v2.py`, `sensory/v4_it.py`** — hierarchia kortykalna (szkielet)
   - Kolejne `CorticalArea` połączone feedforward L2/3→L4 + feedback L5→L2/3 (już wspierane przez `brain_graph.py` delay buffer pattern).
   - V2 ~500 neuronów, V4/IT ~300 (świadomie małe — skalowanie do 2k-10k w Phase 6 po weryfikacji).
   - **Żadnych pretrenowanych featurów.** W tej fazie V4/IT będzie mieć receptive fields odpowiadające surowym statystykom treningowego wideo.
   - Opcjonalny **bootstrap adapter** (`sensory/dinov2_adapter.py`) — ZA FLAGĄ `enable_external_bootstrap=False` domyślnie. Wstrzykuje embeddingi DINOv2 tylko do V4/IT jako dodatkowy afferent (nie zastępuje bottom-up). Dokumentowany jako temporary scaffold do usunięcia.

5. **Active sensing — saccade jako akcja**
   - BG actor z Phase 3 dostaje **dodatkowy head** dla saccade: `(dx, dy)` Δfixation. W dyskretnym wariancie: 9 akcji (8 kierunków + centerowanie).
   - Motor output rozszerza się: `motor_command = {body_action, saccade_action}`.
   - `body_interface.act()` przyjmuje oba; `GridWorldBody` ignoruje saccade, `VisualGridBody` (nowy adapter) renderuje retinalny patch wokół fixation.
   - Saccade reward shaping: information gain w V1 (PE drop) po ruchu oka → positive reward. To jest **Foveated Active Sampling** (Tatler et al. 2011).
   - Reference: Findlay & Walker (1999), Itti & Koch (2001), Tatler et al. (2011).

6. **`sensory/auditory.py`** — równoległy szkielet dla dźwięku
   - **Cochleogram**: mel-filterbank 64 pasma, 1 ms resolution — analog piramidy dla audio.
   - **Auditory attention window**: jeden `(center_freq, Δfreq)` kontrolowany przez BG — analog saccade dla dźwięku.
   - `MGN_adapter` → istniejący thalamus (drugi relay, osobne parametry).
   - `A1` jako `CorticalArea` (~500 neuronów), szkielet.
   - Babbling NIE w tej fazie — to wymaga speech motor + ear-to-mouth delay modeling (Phase 5).
   - Reference: Lyon (2017), Kaas & Hackett (2000).

7. **Embodiment upgrade: `VisualGridBody`**
   - GridWorld z małą teksturą (np. 256×256 bitmapa), agent widzi retinalny patch wokół swojej pozycji.
   - Różne rozdzielczości testowo: 64×64, 256×256, 1024×1024 — **ten sam mózg ma działać** (to weryfikuje scale-invariance).

8. **Testy integracyjne**
   - `tests/test_phase4_retina_scale.py` — retina produkuje tę samą shape outputu dla 64×64 i 1024×1024 inputu.
   - `tests/test_phase4_v1_rf_emergence.py` — po ≥10k kroków na naturalnych obrazach V1 receptive fields mają orientation tuning (mierzone przez SVD na nauczonych wagach L4→L2/3).
   - `tests/test_phase4_saccade_selection.py` — w `VisualGridBody` z celem w losowym miejscu, agent po ≥5k kroków częściej fixuje rejony o high PE niż losowo.
   - `tests/test_phase4_auditory_skeleton.py` — cochleogram + A1 działa end-to-end (nie weryfikujemy jeszcze tonotopii).

### Zakres (co NIE wchodzi)

- **Babbling / speech motor / Wernicke's** — świadomie przesunięte do Phase 5 (wymaga scale-invariant audio najpierw).
- **Somatosensory / touch** — Phase 5+ (wymaga MuJoCo z czujnikami).
- **Interoception poza astrocytą** — Phase 5+.
- **DINOv2 jako główna ścieżka** — tylko opcjonalny adapter, domyślnie off.
- **Strukturalna plastyczność** — nadal odłożona.
- **Concept formation / convergence zones** — emerguje z Phase 4 + Phase 5; nie projektujemy tu eksplicitnie.

### Krytyczne decyzje projektowe (Phase 4)

- **Scale-invariance przez retinę, nie przez warstwy uczone** — piramida + fovea dają fixed-shape input dla reszty mózgu niezależnie od kamery. To jedyne miejsce, gdzie "wymiar obrazu" istnieje.
- **V1 uczona, nie Gabor fixed** — Gabor tylko jako init. Reszta hierarchy również uczona. Żadnych hardcoded feature extractors poza retiną/cochleą (które są biologicznie też fixed).
- **DINOv2 tylko jako scaffold** — uzgodnione z userem. Design-first: najpierw natywna ścieżka, dopiero potem bootstrap jeśli iteracja za wolna.
- **Saccade = mental action via BG** — nie osobny moduł "saliency". BG z Phase 3 to już framework; dodajemy tylko drugi action head.
- **Audio w szkielecie, nie pełny pipeline** — żeby nie powtórzyć błędu plan.md (za szeroki Phase 3). Audio buduje fundamenty dla Phase 5 babbling.
- **Żadnego STFT uczonego przez backprop** — cochlea to biologicznie filtrbank, analog fixed retinal preprocessing.

### Weryfikacja Phase 4

1. **Unit**: `test_retina_output_shape_invariance` — 64×64, 256×256, 1024×1024 inputs → ten sam output shape.
2. **Unit**: `test_gaussian_pyramid_correctness` — piramida ma prawidłowe frequencies (FFT check).
3. **Funkcjonalny**: V1 RF emergence test — orientation tuning index >0.3 dla ≥50% L2/3 neuronów po treningu.
4. **Funkcjonalny**: active saccade test — fixation gęstość koreluje z PE map (Spearman ρ > 0.3).
5. **Funkcjonalny**: pełna ścieżka vision — `VisualGridBody` + `ActionBrain` + retina + V1-V4 uczy się sięgnąć celu, poprawa ≥30% vs Phase 3 baseline (bo ma lepszą reprezentację).
6. **Funkcjonalny**: audio skeleton — cochleogram + A1 + TRN gating przepuszcza puls 1 kHz bez rozlewu aktywności.
7. **Diagnostyczny**: V1 sparsity — population activity ≤10% w stanie ustalonym (lifetime/population sparsity, Willmore et al. 2011).
8. **Diagnostyczny**: saccade frequency 2-4 Hz (biologicznie realistyczne, Rayner 1998).

### Szacowanie złożoności

Pliki nowe: ~10 (`sensory/`, `VisualGridBody`, testy). Pliki modyfikowane: `brain_graph.py` (dwa action heads + nowe relay adapters), `cortex.py` (ewentualnie Gabor init hook). Najtrudniejszy element: **saccade integration w BG** (drugi action head + reward shaping z PE map), bo łączy Phase 3 i Phase 4.

---

## Dependency diagram

```
Phase 2 (MinimalBrain) ──┐
                         ▼
       Phase 3 (ActionBrain + GridWorld + curiosity) ──┐
                                                       ▼
                              Phase 4 (Retina pyramid + V1 STDP + saccade action
                                        + auditory skeleton + VisualGridBody)
                                                       ▼
                              [Phase 5 — babbling, speech, memory, sleep]
```

Phase 3 BLOKUJE Phase 4 (saccade wymaga BG actor). W obrębie Phase 4 retina + V1 + audio mogą iść równolegle; saccade integration na końcu.

---

## Decyzje nadrzędne wynikające z dyskusji

- **Środowisko first**: gym-style abstract (GridWorld/Bandit) → MuJoCo/Unity dopiero po Phase 4.
- **Capability order**: zamknięcie pętli motor → active vision → (babbling w Phase 5). Babbling zostało ZREAKTYWOWANE jako dalszy cel, ale świadomie odsunięte, bo wymaga auditory pipeline który powstaje w Phase 4.
- **Zewnętrzne modele**: dopuszczalne jako opcjonalny adapter za flagą (DINOv2→V4/IT), nigdy jako główna ścieżka. Design natywny najpierw, bootstrap potem.
- **Skalowalność**: rozwiązana przez retinalną piramidę + foveal patch (jedyne miejsce gdzie wymiar obrazu istnieje), nie przez configurowalne warstwy.

---

## Further considerations / otwarte pytania do późniejszej decyzji

1. **Czy BG actor ma być jeden z dwoma headami, czy dwa niezależne BG (motor + saccade)?** Rekomendacja: **jeden actor, dwa heady** — biologicznie BG jest wspólne (dorsomedial vs dorsolateral striatum, ale tu za subtelne). Prostsze do JITa.
2. **Curiosity coefficient β_curiosity adaptacyjne?** Rekomendacja: na stałe 0.1 w Phase 3; adaptacyjne (kontrolowane przez 5-HT) w Phase 5+.
3. **Czy world_model jako inferior olive, czy refactor cerebellum?** Rekomendacja: **world_model**, unikamy duplikacji. Cerebellum dostaje PE jako external input.
4. **Audio sampling rate?** Rekomendacja: 16 kHz dla Phase 4 (wystarczy dla phonemów Phase 5), konfigurowalne.
5. **Czy usunąć całe `arena/` czy zostawić `arena/core.py` ABCs?** Zalecam usunąć całe — ABCs będą naturalne w `embodiment/body_interface.py`. Weryfikacja że nic nie importuje z `arena/` poza testami.
