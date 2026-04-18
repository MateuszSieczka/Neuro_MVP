# Plan: Phase 0 (resurrection + closure) → 5 → 6 → 7 → 8 → 9

## TL;DR

Audyt codebase'u (kwiecień 2026) pokazał, że za czystym frontem JAX/Equinox kryje się **głęboki dług integracyjny**: pięć w pełni zaimplementowanych modułów (`working_memory`, `attention`, `episodic_memory`, `sequence_memory`, `replay_buffer`) jest niewywoływanych poza testami; astrocyta nie jest podpinana do neuronów; oscillator emituje fazy których nikt nie konsumuje; receptor pharmacology działa tylko w BG; free-energy VFE/EFE jest exportowane ale nigdy nie liczone; cała hierarchia ventralna (V2/V4) i auditory (A1) istnieje wyłącznie w testach. Phase 3 nie ma żadnego testu funkcjonalnego.

Ten plan rezygnuje z dodawania nowych dużych modułów dopóki istniejące nie są zintegrowane. **Phase 0** wskrzesza orphans i domyka Phase 3+4 (z testami). Każda dalsza faza dodaje **jedną** nową zdolność kognitywną zbudowaną na zintegrowanych już komponentach. Wszystko biofizyczne (AdEx, STDP, conductance-based); zero ML-owych skrótów (no backprop, no softmax policy, no learned RNN).

Cel końcowy: embodied agent w MuJoCo MJX który (a) widzi i słyszy ze scale-invariant retiną/cochleą, (b) pamięta epizody i konsoliduje je we śnie, (c) tworzy multimodalne koncepty w ATL/konwergencji, (d) gaworzy → uczy się słów → mówi w odpowiedzi → wewnętrznie przemyśla, (e) planuje przez active inference (EFE), (f) utrzymuje cele w PFC.

Każda faza ma trzy progi wejścia do następnej: zielone testy unit, zielone testy funkcjonalne, dwa diagnostyki biofizyczne (firing rates + neuromod balance).

---

## Phase 0 — Resurrection + closure of Phase 3/4

### Cel

Doprowadzić `ActionBrain` do stanu w którym **każdy** moduł `core/` jest albo wywoływany w produkcyjnej pętli, albo świadomie usunięty. Domknąć Phase 3 testowo. Zintegrować wizję end-to-end. Zwolnić "orphaned" warstwy bez wprowadzania nowej funkcjonalności. To jest faza długu technicznego przed dalszą rozbudową.

### Dlaczego najpierw

Każda kolejna faza będzie referowała moduły, które dziś są martwe. Phase 5 zakłada `episodic_memory` i `replay_buffer` (oba orphaned). Phase 6 zakłada cross-modal attention (orphaned). Phase 7 zakłada inner-speech jako PFC `sequence_memory` replay (oba orphaned). Bez Phase 0 każda następna faza buduje na uniwersum z dziurami.

### Zakres

#### 0.1 Diagnostyka kortykalna (gate fazy)

- Test `test_phase0_cortex_alive.py`: `MinimalBrain` na `uniform_dense(0.3)` przez 200 dt → wymagamy `cortex.l5_rate.mean()` ∈ [1, 10] Hz, `l23_state` mean ∈ [2, 12] Hz (Barth & Poulet 2012). Jeśli fail → rekalibracja `psp_target` w `cortex.init_cortical_area_params` przez podniesienie `g_syn_unitary` z 0.43 nS do wartości w której L4 osiąga rheobase przy oczekiwanym afferencie. Kalibracja jest biofizyczna (Feldmeyer 2002 dopuszcza 0.4-1.0 nS), nie taskowa.
- Identyczny test dla `ActionBrain` na `GaussianBanditBody` po 200 cykli decyzyjnych.

#### 0.2 Receptor pharmacology w cortex/thalamus/WM

- Obecnie `CorticalInputs.receptor_gain` defaultuje do 1.0; cortex ignoruje DA/ACh/NE poza efektem przez neuromodulator bus. Naprawa: w `cortex.cortical_area_step` policzyć `LayerModulation = compute_layer_modulation(receptor_params, da, ach, ne, ht)` — funkcja istnieje w `core/receptor.py`. Multiplikatywnie modulować `excitability` L4/L2-3, gain L5 (per `LayerModulation` fields).
- Ten sam pattern dla `thalamus_step` (ACh-gated burst, McCormick 1992 — istnieje pole, nie używane w pełni) i `wm_step` (DA·ACh gating gate już jest, ale brak per-receptor Hill).
- Dodać jeden zestaw `ReceptorParams` per region do `MinimalBrainParams`/`ActionBrainParams`.

#### 0.3 Astrocyte → neurony (active path)

- `core/neuron.py` ma `set_astrocyte()`/`_astrocyte` ale w wersji JAX to martwe pole. Refactor: dodać `astrocyte_state: AstrocyteState | None` do `CorticalAreaParams` (static-shaped bool flag), w `cortex_step` wywołać `astrocyte_step` z `aggregate_to_zones(spike_rates)` i propagować `threshold_shift`/`leak_gain`/`metabolic_lr` do AdEx step (już są policzone w `astrocyte.py`, tylko nie wpinane).
- Wymóg: cortex dostaje JEDNO pole astrocyty (16 zon), shared między L4/L23/L5. WM dostaje swoje. BG swoje. To zmiana Params shape — nie jest to refactor jednego modułu, lecz cross-cutting wiring.
- Po podpięciu: ATP będzie się wyczerpywać przy long sustained activity (zalewy). Gate snu w Phase 5 zacznie być sterowalny.

#### 0.4 Oscillator phase gating

- Faza theta jest emitowana ale nikt jej nie czyta. Wpiąć w cortex_step jako modulator excitability: `cortex_excitability_phase = 1 + 0.1·cos(theta_phase + phase_offset)` (Lakatos et al. 2008 — neuronal oscillations modulate excitability ~10-15%). Niewielki współczynnik, biologicznie udokumentowany.
- W BG dodatkowo: gamma amplitude → striatal MSN burst gating (Berke 2009, ~30 Hz w aktywnym stanie).
- W WM: theta-gamma coupling już zakładany w plan_finished — uczynić explicit przez `wm_step(theta_phase, gamma_amp)`.
- To NIE jest dodawanie funkcjonalności, to konsumpcja sygnału który już istnieje. Brak tego dziś jest bug-like.

#### 0.5 Working memory w ActionBrain (PFC slot)

- WM jest fully tested ale orphaned. Wpięcie: nowy moduł `core/pfc.py` jako thin wrapper łączący jeden `WMState` (32-64 content neurons) z cortexem L2/3 jako goal slot.
- Wiring: `pfc_step` dostaje cortex L2/3 belief jako ff_input + DA + ACh; output `wm_content_rate` projektowany do BG actor jako dodatkowy striatal_drive (Frank & Badre 2012 hierarchical RL — PFC bias na BG).
- Brak goal-encoding logiki; PFC po prostu utrzymuje attractor. Goal-setting (zmiana attractora przez sygnał kontekstowy) wchodzi w Phase 9.
- Reset PFC tylko na `done=True`. Persystencja przez wiele cykli decyzyjnych jest istotą WM.

#### 0.6 Attention module integration

- `attention.py` (Reynolds & Heeger divisive norm + IOR) jest gotowe. Wpiąć jako modulator wejściowych afferentów thalamicznych: `relay.afferent` mnożone przez `attention_output.gain` przed `thalamic_step`.
- Top-down: `attention_step` dostaje `cortex_l5` jako saliency input (cortex steruje na co thalamus puszcza dalej — corticothalamic feedback, Sherman & Guillery 2002).
- IOR uczy się on-line: po fixacji rejonu ACh trace → zmniejszony top-down gain przez `attention_learn`.
- Ten moduł będzie później głównym substratem Phase 7.6 (auditory attention window).

#### 0.7 Sensory stack w ActionBrain (vision end-to-end)

- Nowy moduł `sensory/sensory_stack.py`: pure-functional `SensoryStackParams` = (RetinaConfig, V1Params, V2Params, V4Params, attention_params), `sensory_stack_state`, `sensory_stack_step(state, params, ctx, image, fixation_xy)` → `(new_state, afferent_to_thalamus, v1_pe, v4_belief)`.
- Wewnątrz: retina → lgn_normalize → V1 cortex_step (z Gabor init) → V2 cortex_step → V4 cortex_step. Zwraca `v4_belief` jako "wektor sensoryczny do thalamus_relay" (znacznie krótszy niż surowy LGN ~960 → ~96; biologicznie odpowiada że LGN nie projektuje do całego cortex tylko do V1, V1 do V2 itd.).
- `VisualGridBody._observe` zwraca `image` + `fixation_xy`; `ActionBrain` instancjonuje `SensoryStackState` i wywołuje `sensory_stack_step` przed thalamus_relay.
- Backward-compat flag: `bypass_sensory_stack: bool` static; gdy True ActionBrain bierze `sensory` jak dotąd. Zachowane dla bandit/gridworld testów.

#### 0.8 Saccade info-gain + per-loop credit

- Trzymać `prev_v1_pe` w `ActionBrainState` (skalar = mean V1 L2/3 error).
- Po `sensory_stack_step` policzyć `info_gain = relu(prev_v1_pe - current_v1_pe)` i dodać do `r_total` z `β_saccade = 0.05` (Itti & Baldi 2009 Bayesian surprise).
- Per-loop credit przez **eligibility mask** (nie osobny RPE). Każdy `actor_update` dostaje `eligibility_mask: Array (motor_dim,)` = one-hot ostatniej akcji × `signed_credit_for_this_loop`. Body actor: extrinsic + curiosity + homeostasis. Saccade actor: extrinsic + saccade info-gain. Curiosity i homeostasis mogą iść do obu.
- Decyzja: maska, nie osobny RPE — prostsze, mniej nowego state, biologicznie OK (odpowiada lokalnej D1/D2 plastyczności w specific striatal territory).

#### 0.9 Boredom (PE drift) drive

- W neuromodulator dodać EMA: `pe_long = 0.99·pe_long + 0.01·curiosity`. Sygnał `boredom = relu(pe_long - curiosity)` (PE spadło poniżej długoterminowego średniego → środowisko zbyt przewidywalne).
- `boredom` mapuje na **NE bonus** (lokalnie podnosi LR i temperaturę eksploracji actor-critic — Yu & Dayan 2005). Nie podnosi extrinsic reward.
- Mały drive (waga 0.05). Reference-paper, nie tuned.

#### 0.10 Cerebellum efference copy

- Dziś `cerebellum.mossy` dostaje tylko cortex L5 przez `w_l5_mossy`. Plan_finished wymagał efference. Naprawa: w `_perceive_substep` konkatenować `state.last_body_action` + `state.last_saccade_action` (one-hot) jako dodatkowe mossy afferenty (osobna macierz `w_efference_mossy`).
- Cerebellum będzie się uczyć "gdy ja kazałem akcję A i fovea X, sensory zmieni się tak" — czyli forward model z efference copy (Wolpert 1998), nie tylko z cortex L5.

#### 0.11 Brakujące testy Phase 3+4

- `test_phase3_bandit.py` — 3-armed Gaussian, 800 cykli, p(best) ≥ 0.55. Conservative bo BG dwustopniowy actor jest młody.
- `test_phase3_gridworld.py` — 5×5, 3000 cykli, mean episode length ≥ 20% poniżej random baseline (seed average n=5).
- `test_phase3_curiosity.py` — GridWorld z `goal_reward=0`, 2000 cykli, `curiosity.mean()>0`, body actor entropia akcji > 0.5·log(n_actions).
- `test_phase3_olive_proxy.py` — world_model PE (mean L2 norm) spada ≥30% w 1500 cykli na GridWorld. Test = cerebellum dostaje sensowny sygnał uczenia.
- `test_phase4_v1_emergence_full.py` — 10k kroków VisualGridBody na losowych teksturach, mierzymy orientation-tuning index (Ringach 2002) na L2/3 V1 ≥ 0.3 dla ≥40% neuronów. To jest **długi** test, ale konieczny by uznać V1 STDP za działające.
- `test_phase4_saccade_selection.py` — VisualGridBody 5×5, po 3000 cykli histogram fixation correlates z mapą V1 PE (Spearman ρ > 0.2).
- `test_phase4_end_to_end.py` — VisualGridBody 3×3 + SensoryStack + ActionBrain 2000 cykli, success rate ≥ 0.4 (≥3× random).
- `test_phase0_pfc_persistence.py` — WM utrzymuje attractor po usunięciu drive ≥ 200 dt (memory span > working window).
- `test_phase0_attention_ior.py` — attention_step + attention_learn → po fixacji rejonu jego gain spada o ≥30% (IOR działa).
- `test_phase0_astrocyte_atp.py` — sustained 50 Hz spiking layer → ATP spada do <0.3 atp_max po 5000 dt; threshold_shift > 2 mV.

### Co NIE wchodzi

- MuJoCo, M1 ciągłe, hippocampus refactor, ATL, sleep cycles, language. Wszystkie te są w późniejszych fazach.
- Strukturalna plastyczność (jest w `sparse.py` ale nieużywana — odkładamy do Phase 9+).
- Free energy VFE/EFE callsites (Phase 9 active inference).

### Krytyczne pliki

- `core/cortex.py` — wpiąć `compute_layer_modulation` (+ `astrocyte_step` jeśli `astrocyte_params is not None`), wpiąć `theta_phase` modulację excitability.
- `core/thalamus.py` — wpiąć ACh receptor gain (już jest częściowo, dokończyć).
- `core/working_memory.py` — wpiąć theta-gamma argumenty do `wm_step`.
- `core/pfc.py` — **NOWY** — thin wrapper WM jako goal slot.
- `core/attention.py` — bez zmian; wpinane od strony brain_graph.
- `core/brain_graph.py` — `ActionBrainParams/State` rozszerzone o: `pfc`, `attention`, `astrocyte_cortex`, `astrocyte_wm`, `sensory_stack`, `prev_v1_pe`, `pe_long`, `w_efference_mossy`. `_perceive_substep` rozbudowany o sensory_stack + attention + pfc. `action_brain_step` rozbudowany o saccade info-gain + boredom + per-loop eligibility mask.
- `core/basal_ganglia.actor_update` — przyjmować `eligibility_mask` jako mnożnik credit per-action.
- `sensory/sensory_stack.py` — **NOWY** — pure-functional retina→V1→V2→V4 stack.
- `embodiment/visual_grid.py` — `_observe` zwraca `(image, fixation_xy)` zamiast surowego afferentu (gdy `bypass_sensory_stack=False`).
- `tests/test_phase0_*.py` (3), `tests/test_phase3_*.py` (4), `tests/test_phase4_*.py` (3 nowe).

### Decyzje (Phase 0)

1. **Astrocyta podpięta przez Params, nie przez `set_astrocyte`** — JAX wymaga statycznego pytree. Mutacyjne `set_astrocyte` jest nie-JAX-owe i było zostawione martwe; refactor przez Params jest jedyną poprawną drogą.
2. **PFC jako thin wrapper, nie jako pełna `CorticalArea`** — pełny PFC z 3 warstwami i error neurons jest za duży. Jeden WM slot to prawdziwie minimalny PFC. Cele i hierarchia (Phase 9) dodadzą warstwy.
3. **Eligibility mask, nie per-loop RPE** — rozwiązanie z maską jest prostsze, mniej nowego state, biologicznie OK (lokalna plastyczność striatal territory).
4. **Sensory stack zwraca V4 belief, nie konkat[V4, raw LGN]** — biologicznie thalamus dostaje feedback z cortex hierarchii (pulvinar), nie z LGN bezpośrednio. Symbol "raw LGN do brain" już jest niefizjologiczny. Konkat byłby ML-skrótem.
5. **Boredom sterowany NE (gain), nie reward** — Yu & Dayan 2005: NE = unexpected uncertainty. Boredom to zaskoczenie spadkiem PE (środowisko za nudne), a NE moduluje LR/exploration.
6. **Receptor multiplikatywnie** — Silver 2010, MOD-1 z `neuro_mvp6_fixes.md`. Dodatywne łamałoby niezależność G-białkowych kaskad.

### Weryfikacja Phase 0

1. Wszystkie 4 istniejące testy Phase 4 zielone.
2. Wszystkie nowe testy zielone (3 phase0, 4 phase3, 3 phase4).
3. Diagnostyka biofizyczna (`test_phase0_cortex_alive.py` extension): cortex 1-12 Hz, thalamus 5-15 Hz, BG MSN 1-5 Hz, VTA 3-8 Hz, WM content 5-30 Hz w persistent state.
4. Dopamine bus: phasic spike przy unexpected reward ≥3× tonic (Schultz 1997).
5. ATP balance: pod normalnym taskiem (GridWorld 1000 cycles) ATP nie spada poniżej 0.5; pod sustained drive (`uniform_dense(0.8)`) spada do 0.2-0.3 (przygotowuje gate snu w Phase 5).
6. Manual: `MinimalBrain` JIT-uje się i robi 5000 dt < 10s wall-time po warmup.

---

## Phase 5 — Hippocampal episodic loop + sleep consolidation

### Cel

Dodać **pamięć epizodyczną w pętli** (zapis on-line + recall on-line) i **konsolidację offline** (sen). Nic więcej. M1 ciągłe i MuJoCo trafiają do Phase 6 — łączenie ich z hippocampusem łamałoby zasadę "jedna nowa zdolność per faza".

### Dlaczego teraz

Phase 0 dał Working Memory (krótkie horyzonty) i astrocytę z ATP (substrat snu). Episodic memory uzupełnia hierarchię pamięci o długie horyzonty zdarzeń. Bez tego concept emergence w Phase 7 nie miałaby na czym budować (koncepty = stabilne attraktory powtarzane w replay nocnym).

### Zakres

#### 5.1 Hippocampus jako region

- Nowy moduł `core/hippocampus.py` łączący istniejące `episodic_memory` (DG/CA3 słowniki) + `sequence_memory` (CA3 transition learning) w jeden `HippocampusParams/State`.
- `hippocampus_step(state, params, ctx, ec_in, theta_phase, ne_level)` → `(new_state, ca1_recall, novelty)`.
  - DG sparse coding (już mamy w `episodic_memory.dg_encode`).
  - CA3 pattern completion z sekwencji (rozszerzyć `sequence_memory.seqmem_step` o auto-associative recall — istnieje `predicted_next`, dorobić `pattern_complete_from_partial`).
  - CA1 mismatch: porównanie CA3 recall z aktualnym EC input → mismatch signal do PFC (nowość, Lisman 1999).
- Theta phase gating: encoding na ascending theta (Hasselmo 2002), recall na descending. Implementacja: `encoding_gate = sigmoid(cos(theta_phase) + 0.5)`, `recall_gate = 1 - encoding_gate`.

#### 5.2 EC (entorhinal cortex) jako mikro-most

- HC nie pobiera danych z dowolnego cortex bezpośrednio. Standard: cortex (V4/IT, A1, S1, M1) → EC → HC. Nowa cienka `EntorhinalParams` (jedna `CorticalArea`-lite): konkat głównych cortex outputów → projekcja → DG.
- W Phase 5 sensory ogranicza się do `v4_belief + sensory_raw + last_motor`. Place cells/grid cells (Moser 2008) — odkładamy do Phase 6 razem z MuJoCo (place cells potrzebują ciągłej topologii ruchu).

#### 5.3 Wpięcie HC w ActionBrain

- W `_perceive_substep`: po sensory_stack obliczyć `ec_in`, wywołać `hippocampus_step`. CA1 recall projektowany do PFC (jako dodatkowy ff input do WM); novelty mismatch do `neuromodulator.acetylcholine` (ACh wzrasta przy nowości — Hasselmo 2006, McGaughy 2008).
- Storage w CA3: jeśli `novelty > episodic.ne_threshold` zapisz parę `(ec_in, last_motor_joint)` przez `try_store`. To jest gate na NE (już w `episodic_memory`).

#### 5.4 Replay buffer orchestration

- Plan dotychczasowy zostawił `replay_buffer.py` z komentarzem "orchestration is brain-graph layer's job". Dodać `core/sleep.py` z `sws_replay_step` i `rem_replay_step` używającymi istniejącego `replay_buffer.replay_sample`.
- SWS: reverse replay (chronologically backward). Każdy sampled `Experience` → critic_update + actor_update z czasem skompresowanym (n_substeps=5 zamiast 20). Konsoliduje procedural credit (Wilson & McNaughton 1994).
- REM: forward replay z augmentacją przez world_model rollouts. World model generuje N alternatywnych następników z noise; sequence_memory uczy się tych "snów" (Diekelmann & Born 2010).
- ACh tonic: niskie podczas SWS (umożliwia replay), wysokie podczas REM (Hasselmo 2006). Dodać `sleep_phase ∈ {WAKE, SWS, REM}` do `NeuromodulatorState` jako enum.

#### 5.5 Sleep gate przez astrocytę i oscillator

- Astrocyte ATP < `atp_sleep_threshold = 0.3` → switch oscillator do `slow_wave_mode` (1 Hz, już istnieje w `oscillator.py` jako `sws` flag) → uruchomienie `sws_replay_step` w pętli `brain_graph` poza standardowym `action_brain_step`.
- Brak twardego `done`-triggera. Sen jest endogeniczny i autoregulowany (Achermann & Borbély 2003 process S+C).
- Wyjście ze snu: ATP > 0.8 → switch do WAKE. Mała histereza by uniknąć szybkich oscylacji.
- API: `brain_graph.sleep_cycle(state, params, key, n_swr=200, n_rem=100)` → wywoływane przez run_loop kiedy `state.neuromod.sleep_phase != WAKE`.

#### 5.6 Testy Phase 5

- `test_phase5_dg_separation.py` — dwa podobne stany (cosine 0.9) → dwie różne sparse keys (cosine ≤0.3). Pattern separation działa.
- `test_phase5_ca3_completion.py` — CA3 zaprezentowane z 50% klucza → recall ≥70% sim z pełnym wzorcem.
- `test_phase5_ca1_novelty.py` — sekwencja A,B,C wyuczona; podanie A,X (X≠B) → ACh wzrost ≥2× baseline.
- `test_phase5_sws_consolidation.py` — train na bandyce 200 epizodów + 5 SWS cycles → policy regret na trzymanym test-secie ≤80% baseline (sen pomógł).
- `test_phase5_rem_planning.py` — train world_model na N stanach + REM rollouts → MSE na trzymanych unseen stanach ≤80% bez REM.
- `test_phase5_atp_sleep_gate.py` — sustained activity → ATP ↓ → automatyczne wejście w SWS → ATP regenerates → wyjście do WAKE.
- `test_phase5_theta_phase_encoding.py` — sekwencja inputów na różnych fazach theta → DG zapisuje tylko te na encoding phase.

### Co NIE wchodzi (Phase 5)

- M1 ciągłe, MuJoCo, place/grid cells, ATL, mowa.

### Decyzje (Phase 5)

1. **HC = cienki kompozyt istniejących modułów** — `episodic_memory` i `sequence_memory` są dobrze przetestowane; nie chcemy ich przepisywać. Hippocampus to fasada.
2. **Sen endogeniczny przez ATP** — nie scheduled. To jest jedna z głównych biofizycznych decyzji projektowych z plan_finished. Wreszcie egzekwowalna po Phase 0.3.
3. **EC jako thin wrapper, nie 6-warstwowy cortex** — analogicznie do PFC. Funkcjonalnie potrzebujemy tylko zbiegu modalności do DG.
4. **Brak dreaming z modulacją afektywną** — Phase 7+ (afekt-tagged koncepty są dopiero w Phase 7).

### Weryfikacja Phase 5

1. 7 nowych testów zielone.
2. Phase 0 wszystkie testy nadal zielone (regresja).
3. Diagnostyka: w epizodzie nowej sceny CA1 mismatch >0.2 w pierwszych 5 cyklach, spada poniżej 0.05 po 50.
4. Sleep cycle redukuje skumulowane PE world_model o ≥20% (consolidation).

---

## Phase 6 — Continuous body, M1, MuJoCo

### Cel

Wyjść z dyskretnego GridWorlda do **ciągłego ciała w fizyce**. M1 generuje ciągły torque/velocity command z dyskretnej preparatory activity z BG. Cerebellum dostaje proprioception jako mossy + efference jako climbing. Place cells emergują z ruchu w przestrzeni.

### Dlaczego teraz

Phase 5 dał konsolidację, więc continuous learning na długich trajektoriach (reaching, locomotion) ma gdzie się utrwalać. Phase 7 (multimodal concept emergence) potrzebuje motor jako trzeciego modalu (nie da się "chwycić" konceptu kuli bez motorycznego skojarzenia).

### Zakres

#### 6.1 M1 jako CorticalArea + continuous head

- Nowy `core/m1.py`. M1 = `CorticalArea` (cortex.py) z dodatkowym `motor_readout: Array (n_l5, motor_dim)` mapującym L5 firing rate na continuous joint command.
- BG body actor → M1 preparatory L4 input (jako `td_prediction` w `CorticalInputs`). M1 wykonuje "policy distillation": dyskretna akcja → ciągła trajektoria torque (Wolpert 2001 forward model + inverse).
- `m1_step` zwraca `joint_command: Array (motor_dim,)` jako średnią ważoną L5 firing × motor_readout.
- Uczenie: STDP cortex + Hebbian na motor_readout modulowany RPE z VTA (procedural learning, Doya 2000).

#### 6.2 MuJoCo MJX body

- `embodiment/mjx_arm_body.py`: 7-DOF reacher z `mujoco-mjx` (pure JAX, JIT-able). Sensory: proprioception (joint angles + velocities, 14 dim) + opcjonalnie wizja przez camera RGB → SensoryStack.
- 4 actions BG body actor mapują na "ruch wybranej grupy mięśni" (skrótowo: shoulder_x, shoulder_y, elbow, wrist), M1 ciągle interpoluje.
- Reward extrinsic = -dist(end_effector, target).

#### 6.3 Cerebellum continuous efference

- Phase 0.10 dał efference one-hot do mossy. Tu: pełen `joint_command` (continuous) + proprioception → cerebellum_step. Climbing fiber = `joint_command_actual - joint_command_predicted` (Wolpert 1998 predictor).
- Forward model cerebellum uczy się przewidywać proprioception_t+1 z (proprio_t, joint_command_t).

#### 6.4 Place cells / grid cells

- W EC (Phase 5.2) dodać podpopulację `n_grid_cells = 64` z fixed 2D Gaussian RFs na pozycji ciała w (x,y) (Hafting 2005 z stałymi odstępami sieci heksagonalnej). Hippocampus DG dostaje ich aktywność jako dodatkowy ec_in.
- Pozycja ciała = body kinematics z MuJoCo (czysta funkcja stanu fizyki, nie uczona).
- Place cells emergują w CA3 jako kombinacje grid cells × kontekst sensoryczny — przez normalne CA3 learning, nie hardcoded.

#### 6.5 Testy Phase 6

- `test_phase6_mjx_jit.py` — MJX body + ActionBrain + sensory_stack JIT-uje się i robi 1000 cykli < 30s.
- `test_phase6_reach_learning.py` — 7-DOF reach do statycznego targetu, success rate ≥0.5 po 20k cycles, baseline random ≤0.05.
- `test_phase6_m1_smoothness.py` — M1 generuje smooth torques: jerk metric (3rd derivative L2 norm) < 5× human-typical (Flash & Hogan 1985 minimum-jerk).
- `test_phase6_cerebellum_forward.py` — predicted proprioception MSE spada ≥40% po 5k cycles na reaching.
- `test_phase6_grid_to_place.py` — agent porusza się losowo w gridzie 5×5, po 10k cycles ≥30% CA3 neuronów ma place-field selectivity (firing > 3× baseline tylko w jednej komórce).

### Co NIE wchodzi (Phase 6)

- ATL, multimodal concepts, mowa, gaworzenie. Cała Phase 7.
- Lokomocja (chodzenie) — wymaga modulacji rdzeniowej (CPG) — Phase 9+.
- Manipulacja narzędzi — Phase 9+.

### Decyzje (Phase 6)

1. **MJX, nie PyBullet/Unity** — pure JAX, JIT-uje się z resztą brainem. Plan_finished błędnie sugerował MuJoCo "klasyczny" — MJX jest jego JAX-port, designed właśnie do tego.
2. **M1 z preparatory + continuous head** — Churchland 2012 preparatory subspace. BG nie jest direct motor; M1 jest tłumaczem.
3. **Place cells emergują, grid cells fixed** — grid cells są bardziej "wbudowane" anatomicznie (path integration, Burak & Fiete 2009); place cells są zdecydowanie uczone.

### Weryfikacja Phase 6

1. 5 nowych testów zielone.
2. Phase 0/5 testy zielone (regresja).
3. Diagnostyka: M1 firing rate 5-30 Hz w aktywnym ruchu (Georgopoulos 1986); cerebellum Purkinje 30-100 Hz (Häusser & Clark 1997).
4. End-to-end: ATP cycles (ATP ↓ podczas reaching → sen → ATP ↑) muszą być wykrywalne na 10k+ cycles.

---

## Phase 7 — Multimodal convergence & concept emergence

### Cel

Powiązać modalności (wizja V4 + audio A1 + motor M1 + episodic CA1 + value VTA) w **jednej regionie konwergencji** (ATL/AG hub). Dla embodied AGI to substrat **konceptów**: stabilne attraktory uruchamiane przez dowolny modal danego doświadczenia.

### Dlaczego teraz

Phase 6 dał motor i fizykę. Phase 5 dał epizody. Bez ATL agent ma odseparowane representacje per-modal i żaden mechanizm nie wiąże ich w "tę samą rzecz". To jest _crucial step_ do AGI.

### Zakres

#### 7.1 Auditory pipeline w ActionBrain

- A1 dotąd standalone. Wpięcie analogiczne do SensoryStack vision: nowy `sensory/auditory_stack.py` (cochlea → MGN normalize → A1 cortex → opcjonalnie A2/STG jako kolejna `CorticalArea`).
- ActionBrain dostaje drugi sensory port: `audio_waveform` przekazywane przez auditory_stack do drugiego thalamic relay (MGN), drugiego cortex (A1), output → ATL.
- Brak nowych BG actors w tej fazie (auditory attention window — Phase 8 razem z mową).

#### 7.2 ATL convergence zone

- Nowy `core/atl.py` jako `CorticalArea` (~512 neurons) z afferentami konkatenowanymi: `[v4_belief, a1_belief, m1_l5, ca1_recall, vta_value_tag]`.
- Slow STDP: niskie `learning_rate` (1/10 cortexu standardowego), długi integration window. Cel: stabilne attraktory.
- Lateral inhibition silne (k-WTA, k=5%): konkurencja → kategorie.
- Synaptic scaling (Turrigiano 2008): per-neuron rate homeostasis (już w `plasticity.py` jako homeostatic_scaling — wpiąć tutaj jeśli orphaned, jak attention).
- Reference: Patterson, Nestor, Rogers (2007) hub-and-spoke; Binder & Desai (2011).

#### 7.3 Cross-modal novelty drive

- Curiosity dotąd była per-cortex PE. Tu nowy sygnał: `cross_modal_pe = ||predicted_audio_from_atl - actual_audio||` (i analogicznie dla innych par). Predykcje generuje ATL przez generative weights do każdego modalu (PC top-down feedback do V4, A1).
- Wysokie cross_modal_pe → silne ACh + curiosity bonus do reward. Generuje zachowanie "patrz i słuchaj" + powtarzane investigation.

#### 7.4 Value tagging

- VTA RPE → krótka okno (200 dt) eligibility w ATL: neurony aktywne tuż przed dużym RPE dostają wzmocnienie wagowych projekcji do `affective_field` w ATL (subset 64 neuronów reprezentujących valence).
- Reference: Pessoa (2008) emocje jako value-tagged concepts, LeDoux (2000) afektywne tagging.

#### 7.5 Multimodal task environment

- `embodiment/multimodal_grid.py`: VisualGridBody + per-cell tone. Konsystentne pary (color, freq) w 80% komórek, niekonsystentne 20% jako noise.
- Nowy `embodiment/object_world.py` z MJX (Phase 6+): kilka obiektów z teksturami i nazwami-tonami, agent może podejść, dotknąć, popchnąć.

#### 7.6 Testy Phase 7

- `test_phase7_atl_selectivity.py` — po 15k cycles na multimodal_grid ≥30% ATL neuronów ma d' > 1.5 dla par (V+A) konsystentnych vs niekonsystentnych.
- `test_phase7_cross_modal_recall.py` — ATL aktywuje się przy podaniu tylko V; predicted_audio z ATL ≥ 0.6 cosine z faktycznym tone tej kategorii.
- `test_phase7_concept_stability.py` — 3 epizody odstępu 24h-symulowane (tj. 3 sleep cycles między) → koncept reaktywuje się z całym modal-agnostic profile (CA1 recall + ATL recall + V4 PE drop).
- `test_phase7_value_tagging.py` — koncepty skojarzone z reward mają baseline rate ≥1.5× neutralnych.
- `test_phase7_audio_in_loop.py` — pure tone 1kHz w loop, ActionBrain JIT-uje, A1 firing rate 2-15 Hz.

### Co NIE wchodzi (Phase 7)

- Mowa (gaworzenie/produkcja/rozumienie). Phase 8.
- Inner speech. Phase 8.
- Compositional concepts. Phase 9.

### Decyzje (Phase 7)

1. **Jeden ATL zamiast hub-and-spoke z multiple convergence sites** — biologicznie uproszczenie (Patterson 2007 sam mówi że ATL jest dominującym hubem).
2. **Slow STDP + synaptic scaling** — koncepty muszą być stabilne na timescale wielu epizodów. Standard cortex STDP jest za szybki.
3. **Generative top-down z ATL do każdego modalu** — Rao & Ballard 1999. To umożliwia cross_modal_pe i imagination ("widzę kulę → przewiduję dźwięk").
4. **Value tag jako subset ATL, nie osobny region** — amygdala jako pełny region jest Phase 9+. Tu: lekkie afektywne tagging w ATL.

### Weryfikacja Phase 7

1. 5 testów zielone. Regresja Phase 0/5/6.
2. Diagnostyka: ATL stable attractor — dwa wejścia o cosine 0.8 → ATL output cosine ≥0.95.

---

## Phase 8 — Speech, babbling, comprehension, inner thought

### Cel

**Mowa.** Najtrudniejsza faza. Wymaga: (a) speech motor M1 + vocal tract, (b) feedback przez własne ucho (ear-mouth loop), (c) Wernicke (comprehension), (d) inner speech jako PFC sequence replay z motor suppression.

### Dlaczego teraz

Phase 7 dał koncepty. Język to sekwencjonowanie konceptów do artykulacji. Bez konceptów językowych nie ma podstawy znaczeniowej (czysto syntaktyczny SNN to chińskie pokoje).

### Zakres

#### 8.1 Speech motor M1 (laryngeal)

- Drugi M1 head: `m1_speech` z motor_dim ≈ 30 (artikulatory: jaw, lip rounding, tongue front/back/height, larynx pitch, voicing). Może być kolejnym M1 lub osobnym head z tej samej `CorticalArea`.
- Decyzja: **osobny moduł** `core/m1_speech.py` z osobnymi STDP, bo timescale i target (kontynualny waveform) są inne niż reaching.

#### 8.2 Vocal tract synthesizer

- `embodiment/vocal_tract.py`: deterministyczny Klatt 1980 cascade-parallel formant synthesizer (lub Pink Trombone-style). Input: 30 DOF z `m1_speech`. Output: waveform 16 kHz.
- Waveform → `auditory_stack` (Phase 7) → A1 → ATL. **Self-feedback loop zamknięta.**

#### 8.3 Babbling

- Pre-training 30k cycles bez task: M1_speech random init, agent słyszy siebie. STDP w M1_speech ↔ A1 powiązanie ulega (article→sound mapping).
- Curiosity drive: info-gain w A1 po artykulacji → preferencja dla wokalizacji o maksymalnej różnorodności (analog saccade info-gain z Phase 0).
- Reference: Guenther DIVA model (2016), Oller 1980 canonical babbling, Kuhl 2004 perceptual magnet.

#### 8.4 Wernicke (comprehension)

- Nowy `core/wernicke.py` jako `CorticalArea` po A1 (w auditory hierarchii). Sequence_memory wpięte tu (Phase 0 zostawił seq-mem orphaned w hippocampus; tu drugie callsite — to OK, sequence_memory to general primitive).
- Słyszę "kula" → A1 sekwencja → Wernicke uczy się "to słowo to klasterujące pattern" → projekcja do ATL → odpala koncept "kula".
- Wernicke → ATL projekcja jest plastyczna (Hebbian), uczy się collocation słowo-koncept.

#### 8.5 Productive speech (concept → articulation)

- Reverse path: ATL koncept → PFC sequence_memory (Phase 0 PFC jako goal slot, tutaj rozszerzamy o sequence_memory wewnątrz PFC) → projekcja do M1_speech → vocal_tract.
- Tutor environment `embodiment/tutor_env.py`: opiekun pokazuje obiekt + wymawia słowo; agent próbuje powtórzyć; reward = cosine podobieństwo waveform-cochleogram (DTW), Kuhl 2004 social gating.

#### 8.6 Auditory attention (3rd BG loop)

- Trzeci aktor BG: `actor_audio_attention` (motor_dim = 8: 4 kierunki frequency × 2 czasowe). Wybiera okno spektrogramu do uwagi (analog saccade dla wzroku).
- Skaluje per-loop credit z Phase 0 do trójki: body / saccade / audio_attend.

#### 8.7 Inner speech / thought

- W PFC dodać `motor_suppress: bool` field. Kiedy `True`: sequence_memory replay aktywuje M1_speech ALE `vocal_tract.act` nie jest wywoływane (efference bez motor output).
- Cerebellum forward model przewiduje "co bym usłyszał" → predicted cochleogram → A1 → ATL — jakbym mówił, ale cicho.
- Thought chain: ATL koncept A → PFC seq → koncept B → seq → koncept C... iteracyjnie, każda iteracja = jeden cykl decyzyjny z PFC autoreplay.
- Reference: Vygotsky 1934, Hurlburt & Heavey 2006, Alderson-Day & Fernyhough 2015.

#### 8.8 Testy Phase 8

- `test_phase8_babbling_diversity.py` — po 30k cycles ≥10 klastrów spectralnych w produkowanych wokalizacjach (k-means na cochleogram).
- `test_phase8_word_learning.py` — po 5k tutor interactions, agent reproducuje target z DTW similarity ≥0.6.
- `test_phase8_comprehension.py` — słysząc "kula" → fixacja na obiekcie kula w 5-obiektowym MJX environments ≥70% w 8 sekund.
- `test_phase8_inner_speech.py` — PFC autoreplay z motor_suppress generuje sekwencję ATL aktywacji bez nonzero vocal_tract output.
- `test_phase8_thought_chain.py` — startując od konceptu A, sekwencja długości ≥4 z co najmniej 2 unique concept transitions niezdegenerowana w pętli A→A→A.
- `test_phase8_aud_attention.py` — w polu z dwoma równoczesnymi tonami (450, 1100 Hz) agent po 2k cycles preferuje fixację na jednym (entropia rozkładu fixacji < 0.5 max).

### Co NIE wchodzi (Phase 8)

- Gramatyka wielowyrazowa, składnia rekurencyjna — Phase 9.
- ToM, intencje innych, wnioskowanie społeczne — Phase 9+.
- Czytanie/pisanie — out of scope.

### Decyzje (Phase 8)

1. **Klatt synthesizer, nie WaveNet** — biofizyka niż ML. Klatt jest deterministyczny i ma dokładnie te DOF jakie potrzebujemy.
2. **PFC z sequence_memory wewnątrz** — sequence_memory to "goal sequencer" w PFC. Hippocampus uczy zdarzeniowych sekwencji; PFC uczy sekwencji intencjonalnych.
3. **Inner speech = motor_suppress + cerebellum re-afference** — to jest _the_ biologiczna teoria wewnętrznej mowy (Tian & Poeppel 2010 "mental imagery of speech").
4. **Audio attention jako trzeci aktor BG** — kontynuacja parallel-loops architektury (Alexander 1986). Skaluje się.

### Weryfikacja Phase 8

1. 6 testów zielone.
2. Regresja wszystkich poprzednich faz.
3. Diagnostyka: M1_speech firing 10-50 Hz w aktywnej artykulacji; baseline 1-3 Hz.
4. ACh poziom: wysoki podczas comprehension, średni podczas tutor reward, niski podczas inner speech (Hasselmo 2006).

---

## Phase 9 — Hierarchical control, active inference, ToM (open-ended)

### Cel (szkic)

Po Phase 8 agent ma podstawowy embodied AGI substrat: ciało, percepcja, pamięć, sen, koncepty, mowa, myśl. Phase 9+ to:

- **Goal stack w PFC** (hierarchiczne RL, Botvinick 2014).
- **Aktywne wnioskowanie z EFE** — wreszcie call-site dla `expected_free_energy()` z `core/free_energy.py`.
- **Strukturalna plastyczność** — call-site dla `synaptogenesis()/prune_below()` z `core/sparse.py`.
- **Theory of Mind** — drugi agent w środowisku, koncepty na temat jego stanów.
- **Compositional language** — sequence_memory hierarchiczny (jeden poziom słów, drugi fraz).
- **Tool use** — extension manipulation w MJX.

Każdy z tych elementów to osobna faza ~rozmiaru Phase 5-7. Teraz są tylko placeholderem; konkretny plan Phase 9.x napiszemy po Phase 8.

---

## Dependency graph

```
Phase 0 (resurrection + closure: PFC, attention, astrocyte, oscillator, sensory_stack, V1 emergence, saccade info-gain, Phase 3 testy)
   ↓ blokuje
Phase 5 (HC + EC + replay orchestration + sleep-by-ATP)
   ↓ blokuje
Phase 6 (M1 continuous + MJX + cerebellum efference + place/grid)
   ↓ blokuje
Phase 7 (auditory in loop + ATL + cross-modal novelty + value tagging)
   ↓ blokuje
Phase 8 (M1_speech + vocal_tract + babbling + Wernicke + inner speech + 3rd BG loop)
   ↓
Phase 9+ (goal stack, EFE, structural plasticity, ToM, grammar, tools)
```

Phase 0 jest wąskim gardłem — bez wskrzeszenia orphans nie ma jak wywołać `episodic_memory` w Phase 5, `attention` w Phase 7, `sequence_memory` w Phase 8.

---

## Cross-cutting decyzje (architektura)

1. **Wszystko biofizyczne, zero ML hacków**. Konkretnie zakazane: backprop, softmax-policy, learned RNN gates, Adam, learned positional embeddings, normalizing flows, attention z dot-product key-query (jest nasza biologiczna divisive-normalization attention!).
2. **Każdy moduł albo w produkcji albo usunięty**. Po Phase 0 nie tolerujemy orphans. Audyt bowiem pokazał, że orphans rosną szybciej niż integracja.
3. **Astrocyta wszędzie tam gdzie są spike-bursty neurons** — cortex, BG, WM, M1, A1, ATL. To koszt JIT + Params shape, ale daje konsystentny ATP/precision system.
4. **Receptor pharmacology w każdej regionie** — nie tylko BG. Cortex, thalamus, WM, ATL muszą mieć `compute_layer_modulation` calls.
5. **Sequence_memory jest re-używalny** — w hippocampus (CA3) i w PFC (intencje). Dwa callsite, ten sam moduł.
6. **Attention jest re-używalna** — wpięta w thalamus (przed cortex), później w ATL (gating sensory dla konceptów), później w Wernicke (selektor dźwięku).
7. **Per-loop BG actors skaluje się** — refactor `ActionBrain` na `actors: List[ActorParams]` z masking-based credit już w Phase 0. Dodawanie pętli (saccade, audio attend, etc.) jest wtedy trywialne.
8. **Brak external pretrained models w głównej ścieżce** — DINOv2/CLIP/Whisper itd. dopuszczalne tylko jako opcjonalny `bootstrap_adapter` za flagą domyślnie False, jak ustalono w plan_finished. Phase 0-9 nie używają żadnego.

---

## Open questions (do rozstrzygnięcia w trakcie)

1. **PFC: jeden WM slot czy multiple (Cowan 4±1)?** — Phase 0 startuje z jednym (minimum żywotne); rozważyć multiple w Phase 5 jako prerequisite dla goal stack w Phase 9.
2. **Astrocyta zone count per region** — domyślnie 16 (`astrocyte.AstrocyteParams.n_zones`); może wymagać per-region override (cortex większe, BG mniejsze).
3. **Sleep cycle długość** — 200 SWS + 100 REM to estymata; kalibrować empirycznie po Phase 5.
4. **Vocal tract: Klatt vs Pink Trombone** — Klatt deterministyczny, Pink Trombone bardziej intuicyjny dla ssania. Decyzja: Klatt (tradycja phonetic synthesis).
5. **MJX vs MuJoCo Python bindings** — MJX (pure JAX), confirmed.
6. **Czy dodawać amygdala jako region** — Phase 7 robi value tagging w ATL subset; pełna amygdala (LeDoux) Phase 9+.
7. **Czy zostawić bandit/gridworld po Phase 6** — TAK, jako fast smoke-tests dla kolejnych refaktorów. Tylko visual_grid migruje do MJX object_world.

---

## Co wreszcie zaowocuje (po Phase 8)

Embodied agent w MJX który:

- ma stabilne _koncepty_ (ATL attractors aktywowane przez dowolny modal),
- _pamięta epizody_ (HC), _konsoliduje je w nocy_ (SWS+REM),
- _patrzy aktywnie_ (saccade info-gain), _słucha aktywnie_ (audio attention),
- _manipuluje ramieniem_ (MJX + M1 + cerebellum forward model),
- _gaworzy_, _uczy się słów od opiekuna_, _mówi w odpowiedzi na koncept_,
- _myśli wewnętrznie_ (PFC sequence replay z motor suppression),
- _dryfuje między snem a czuwaniem endogenicznie_ (ATP-driven),
- nigdy nie używa backprop, softmax policy ani żadnego ML-shortcuta.

To jest baseline AGI-substrate. Phase 9+ to dopisywanie kompetencji wyższego rzędu (gramatyka, ToM, narzędzia, planowanie z EFE).
