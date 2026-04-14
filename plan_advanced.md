# PLAN ZAAWANSOWANY: SNN → Fundament AGI (następca plan.md, kontynuacja plan_base.md)

## TL;DR

plan_base.md (Fazy 2-8) eliminuje hacki RL/ML i uzupełnia brakującą biofizykę neuronalną. Niniejszy plan kontynuuje od Fazy 9: wprowadza topologię sparse, dynamikę synaptyczną, pamięć asocjacyjną, konsolidację wiedzy, wiązanie fazowe, bramkowanie wzgórzowe i planowanie atraktorowe. Każdy koncept z plan.md Faz 2-5 jest oceniony: ACCEPT / MODIFY / DEFER / REJECT.

---

## AUDYT KONCEPTÓW Z PLAN.MD FAZ 2-5

### Status każdego konceptu:

| # | Koncept | plan.md ref | Werdykt | Uzasadnienie |
|---|---------|------------|---------|--------------|
| 2.1 | Small-World Connectivity | Faza 2, Krok 2.1 | **ACCEPT** | O(N²) to fundamentalna blokada skalowania. Watts-Strogatz 1998 rock-solid. |
| 2.2 | Structural Plasticity | Faza 2, Krok 2.2 | **MODIFY** | Najpierw pruning (prosty), synaptogeneza później. G_max budget konieczny. |
| 2.3 | Kanerva SDM | Faza 2, Krok 2.3 | **MODIFY** | DG expansion już istnieje. Zamienić recall (cosine→matrix multiply). Zachować NE-gated storage. |
| 2.4 | EWC / PKMzeta | Faza 2, Krok 2.4 | **ACCEPT** | Brak ochrony wiedzy = catastrophic forgetting. Fisher diagonal z eligibility proxy. |
| 2.5 | VRP (Vesicle Release) | Faza 2, Krok 2.5 | **ACCEPT** | Zastępuje hack `_vesicle_p=0.8`. STD/STF emergentne. Naturalna regularyzacja. |
| 3.1 | Phase Binding (Synchrony) | Faza 3, Krok 3.1 | **MODIFY** | Koncept correct, implementacja uproszczona: gamma-phase gated propagation zamiast full arrival-time correction. |
| 3.2 | Attention Circuit | Faza 3, Krok 3.2 | **MERGE** | Pokrywa się z plan_base.md HACK D. Mutual-inhibition circuit to konkretna implementacja. Wykonać w plan_base Phase 6. |
| 3.3 | Remove np.random.choice | Faza 3, Krok 3.3 | **DONE** | WTA w D1D2Actor zaimplementowane. Softmax usunięty z action selection. |
| 3.4 | 1000 Brains / Grid Cells | Faza 3, Krok 3.4 | **DEFER** | Wymaga środowisk przestrzennych (obrazy, nawigacja). Arena ma tylko 1D discrete. |
| 3.5 | Homeostatic Drive (Anti-Dark-Room) | Faza 3, Krok 3.5 | **DEFER** | Sensowny dopiero w open-ended environments. Obecne arena environments zawsze dają sygnał reward. |
| 4.1 | Thalamic Gating | Faza 4, Krok 4.1 | **ACCEPT** | Czyste rozdzielenie percepcja/wyobraźnia. ACh-gated learning kluczowy. |
| 4.2 | Wave Planning (Attractor Drift) | Faza 4, Krok 4.2 | **MODIFY** | Theta-sweep w network.py już implementuje multiplexed evaluation. Rozszerzyć o thalamic gate zamiast budować od zera. |
| 4.3 | Hierarchical Delegation | Faza 4, Krok 4.3 | **DEFER** | HierarchicalSequenceMemory istnieje (3 levels). BG veto/approve wymaga złożonych zadań wielopoziomowych. |
| 4.4 | SWR Consolidation | Faza 4, Krok 4.4 | **DONE** | replay_buffer.py ma pełne SWS/REM z oscylator-gated Up/Down, VTA RPE, three-factor STDP. |
| 4.5 | Energy-Triggered Sleep | Faza 4, Krok 4.5 | **ACCEPT** | Obecny trigger `done=True` łamie P2 (ciągłość). ATP threshold jest termodynamicznie poprawny. |
| 5.1-5.5 | Integration Tests | Faza 5 | **ACCEPT** | Brak testów end-to-end z continual learning. Krytyczne dla walidacji EWC. |

---

## CO JUŻ JEST ZROBIONE (z plan.md Faz 2-5)

1. **WTA action selection** (3.3) — D1D2Actor używa spike count / voltage evidence + argmax. Softmax + np.random.choice usunięte.
2. **SWR sleep consolidation** (4.4) — replay_buffer.py: SWS reverse replay + REM forward replay, oscylator Up/Down gating, VTA RPE, three-factor STDP, seizure brake.
3. **SWS/awake oscillator mode** (4.4 partial) — oscillator.py: `enter_sws()` / `exit_sws()` z ~1Hz slow oscillation, Up/Down state transitions.
4. **Hierarchical sequence memory** (4.3 partial) — sequence_memory.py: 3-level hierarchy (gamma/theta/episode), theta-phase gating, DG expansion.
5. **DG sparse expansion** (2.3 partial) — episodic_memory.py: random projection + competitive k-WTA sparsity.
6. **Theta-sweep planning** (4.2 partial) — network.py: `theta_sweep_plan()` z efference copy i multiplexed evaluation.
7. **Precision-weighted feedback** (4.1 partial) — network.py: top-down predictions scaled by inverse PE variance.

---

## CO NALEŻY ODRZUCIĆ / ODŁOŻYĆ

### REJECT: Brak konceptów z planu do odrzucenia permanentnie.
Wszystkie koncepty z plan.md Faz 2-5 są biofizycznie uzasadnione. Żaden nie jest hackiem.

### DEFER (odłożone do kolejnego planu):

#### 1. 1000 Brains / Grid Cells (plan.md 3.4)

**Powód:** Grid cells wymagają przestrzeni ciągłej do tracking pozycji (displacement integration). Arena environments (SingleButton, TwoButton, Corridor, ShiftingBandit, TMaze) operują na 1D-4D state space bez pojęcia "nawigacji". Grid cells nie mają czego śledzić.

**Warunek powrotu:** Dodanie środowisk 2D nawigacyjnych (gridworld, maze z ciągłą pozycją). Wtedy grid cell modules mogą śledzić displacement, a kolumny korowe budować object-centric reference frames.

**Co zachować z konceptu:** Idea multi-scale periodycznego kodowania jest wartościowa. Jeśli input_dim wzrośnie >64, rozważyć grid-cell-like encoding zamiast Gaussian population encoding.

#### 2. Homeostatic Drive / Anti-Dark-Room (plan.md 3.5)

**Powód:** `HomeostaticDrive` z `arousal_setpoint` i `energy_setpoint` generuje interoceptive PE wymuszający aktywność. Ale w obecnych environments agent zawsze dostaje reward signal — nie ma możliwości "dark room" (zerowy input). Mechanizm jest biologicznie poprawny ale testowo bezużyteczny.

**Warunek powrotu:** Open-ended environment z opcją "nic nie robienie" bez penality. Albo: continual learning z przerwami (brak input przez sekundy).

**Co zachować z konceptu:** ATP-driven sleep (4.5) realizuje CZĘŚĆ tego konceptu — wyczerpanie energii wymusza sen/regenerację. `free_energy.py` ma placeholder na priory interoceptywne.

#### 3. Hierarchical Delegation / BG Veto-Approve (plan.md 4.3)

**Powód:** HierarchicalSequenceMemory już generuje predictions na 3 skalach. BG veto/approve macro-goals wymaga zadań z wyraźną hierarchią celów (złożone sekwencje, subgoals). TMaze jest jedynym kandidatem, ale jest zbyt prosty.

**Warunek powrotu:** Zadania z >2 podcelami sekwencyjnymi (np. klucz → drzwi → skarb).

#### 4. Full Variational Inference (nie w plan.md, ale w scope)

**Powód:** free_energy.py to skeleton z F = ½Σπε². Pełne MCMC/message-passing wymaga generative model p(o|s) i recognition model q(s|o). Predictive coding z relaxation loop (plan_base BIO 7) jest wystarczającym proxy na obecną skalę.

---

## FAZY IMPLEMENTACJI (kontynuacja plan_base.md)

### Faza 9: Sparse Topology & Axonal Delays

*Zależności: plan_base.md Phase 3 (dual-exponential synapses) i Phase 7 (cleanup) zakończone.*

Cel: Przejście z O(N²) dense macierzy na O(N×k) sparse Small-World. Umożliwia skalowanie do >1000 neuronów i wprowadza biologicznie poprawne opóźnienia aksonalne.

#### 9.1: SmallWorldConnectivity class

Nowa klasa w `core/synapse.py` (rozszerzenie istniejącego modułu). Watts-Strogatz (1998) generuje graf z:
- `k_local=6` lokalnych sąsiadów per stronę (degree=12)
- `p_rewire=0.1` prawdopodobieństwo przeokablowania → "autostrady"

**Struktura danych:** Structure of Arrays (SoA): `pre_idx, post_idx, weights` — trzy wektory 1D NumPy.

**Operacja forward:** `np.bincount(post_idx, weights=pre_spikes[pre_idx] * weights, minlength=n_post)` — w pełni wektoryzowane O(E) zamiast O(N²). UWAGA: NIE używać `np.add.at()` (nie korzysta z SIMD).

**Opóźnienia aksonalne per-synapsa:** Czwarty wektor `delays` w SoA.
- Lokalne krawędzie: `delay = local_delay_ms` (1.0 ms)
- Rewired "autostrady": `delay = d_ring × long_range_delay_factor` (0.5 ms/hop)
- Delay buffer: `_output_history` deque (identyczny wzorzec jak w NetworkGraph)

**Dodać do `core/config.py`:** `TopologyConfig(BaseConfig)` z polami: `k_local`, `p_rewire`, `use_sparse`, `local_delay_ms`, `long_range_delay_factor`.

**Integracja:** `AdExLayer` i potomne klasy: jeśli TopologyConfig podane → `self.connectivity = SmallWorldConnectivity(...)` zamiast gęstego `self.w`. `forward()`: `current = self.connectivity.multiply(pre_spikes)`. STDP: operuje na sparse `connectivity.weights`.

**Gwarancja spójności:** Po inicjalizacji i po każdym cyklu structural plasticity: `assert min(in_degree) > 0 and min(out_degree) > 0`. Jeśli orphaned → fallback to ring neighbor.

**Pliki:** `core/synapse.py` (SmallWorldConnectivity), `core/config.py` (TopologyConfig), `core/neuron.py` (AdExLayer integracja)

#### 9.2: Migracja istniejących warstw na sparse

Wszystkie warstwy z dense `w`:
- `SNNDeepCritic.w_h` — critic feedforward
- `D1D2Actor.w_d1, w_d2` — actor feedforward
- `PredictiveCodingLayer` — PC weights
- `InhibitoryPool.w_ei, w_ie` — E→I i I→E

Strategia: backward-compatible. Jeśli `topology_cfg is None` → dense (obecne zachowanie). Jeśli podane → sparse. Pozwala na stopniową migrację i A/B testing.

**Pliki:** `core/basal_ganglia.py`, `core/neuron.py`, `core/interneuron.py`, `core/predictive_coding.py`

**Weryfikacja:**
- Graf 1000 neuronów, k=6, p=0.1: clustering C > 0.3, avg path < 10 (vs C≈0 i L≈500 dla random graph)
- Test pamięci: O(N×k) zamiast O(N²)
- Regresja: identyczne wyniki na SingleButton/TwoButton z sparse vs dense (same seed, same initial weights)

---

### Faza 10: Synaptic Microphysics — Vesicle Release Probability

*Zależności: Faza 9 (sparse topology dostępna do per-synapse wektoryzacji).*

Cel: Wprowadzenie stochastycznej dynamiki pęcherzykowej (Zucker & Regehr 2002) jako naturalnej regularyzacji i źródła biologicznego szumu. Zastępuje hack `_vesicle_p=0.8` w world_model.py.

#### 10.1: VRP (Vesicle Release Probability) w SynapticChannels

**Dodać do `core/config.py`:** `VRPConfig(BaseConfig)` z polami:
- `n_docked_max: int = 10` — max pęcherzyków w docked pool
- `p_release_base: float = 0.3` — bazowe p_release (Branco & Staras 2009: cortical synapses ~0.2-0.4)
- `tau_recovery: float = 800.0` (ms, Zucker & Regehr 2002)
- `tau_facilitation: float = 200.0` (ms)
- `facilitation_increment: float = 0.05`

**Stan per-synapsa (wektoryzowany):**
- `n_docked: NDArray[float32]` — pęcherzyki gotowe
- `p_release: NDArray[float32]` — bieżące prawdopodobieństwo

**Na pre-spike:**
```
released = np.random.binomial(n_docked.astype(int), p_release)
n_docked -= released
effective_weight = weight * released / n_docked_max
p_release += facilitation_increment * (1 - p_release)  # STF
```

**Między spike'ami:**
```
n_docked += (n_docked_max - n_docked) * (1 - exp(-dt/tau_recovery))  # recovery
p_release = p_release * exp(-dt/tau_facilitation) + p_release_base * (1 - exp(-dt/tau_facilitation))  # decay
```

**Emergentne efekty:**
- **Short-term depression (STD):** częste spike'i → n_docked spada → synapsa słabnie
- **Short-term facilitation (STF):** seria spike'ów → p_release rośnie → następne silniejsze
- **Naturalny dropout:** stochastyczność Binomial(n,p) → regularyzacja bez sztucznych flag

#### 10.2: Usunięcie world_model vesicle hack

`SNNWorldModel._vesicle_p = 0.8` i `_decode_with_vesicle_noise()` z Bernoulli mask → USUNĄĆ. VRP w SynapticChannels daje ten sam efekt (stochastyczne uwolnienie) na poziomie synapsy, nie dekodera.

**Pliki:** `core/synapse.py` (VRP w SynapticChannels), `core/config.py` (VRPConfig), `core/world_model.py` (usunięcie _vesicle_p hack)

**Weryfikacja:**
- STD: pre spike train 100Hz × 500ms → effective weight < 30% po 200ms, recovery w ~1s
- STF: paired-pulse ratio > 1.0 przy ISI=50ms
- Brak `_vesicle_p` w codebase: `grep -rn "vesicle_p" core/` → 0

---

### Faza 11: Pamięć Asocjacyjna — Kanerva SDM

*Zależności: Faza 9 (sparse topologia dla efektywnej pamięci).*

Cel: Zamienić O(N) cosine similarity search w episodic_memory.py na O(1) matrix-multiply recall (Kanerva 1988). DG sparse expansion (Address Decoder) już istnieje — brakuje tylko Content Matrix (CA3).

#### 11.1: Content Matrix (CA3)

Obecna `EpisodicMemory` przechowuje `_episodes: list[Episode]` i `_keys: list[NDArray]`. Recall = iteracyjny `cosine_similarity(cue, key)` → O(N).

**Transformacja:**
- USUNĄĆ: `_episodes: list[Episode]`, `_keys: list[NDArray]`, pętlę cosine similarity
- DODAĆ: `W_content: NDArray[float32]` shape `(dg_dim, content_dim)`
  - `content_dim = state_dim + 1 + state_dim` (state + reward + next_state)

**Zapis (Hebbian):**
```
a = self._dg_encode(state)  # sparse binary address (istniejące)
content = np.concatenate([state, [reward], next_state])
W_content += np.outer(a, content)  # Hebbian outer product
```

**Odczyt (O(1)):**
```
a = self._dg_encode(cue)
recalled = a @ W_content / max(np.sum(a), 1)  # normalized by active addresses
```

**Pojemność:** SDM z M=dg_dim i sparsity p=0.05: pojemność ∝ exp(M × H(p)) ≈ eksponencjalna.
Dla M=320 (64 × 5 expansion): ~10⁴ wzorców (vs 500 fixed capacity obecny).

#### 11.2: Decay i normalizacja

Bez decay `W_content` rośnie nieograniczenie.
- Per step: `W_content *= (1 - sdm_decay_rate)` z τ → `sdm_decay_rate = 1e-5`
- Po zapisie: normalizacja aktywnych rzędów przez max norm
- Biologicznie: synapsy CA3 podlegają turnover

#### 11.3: Zachować istniejące mechanizmy

- **NE-gated storage** — `try_store()` z `ne_level >= ne_threshold` → BEZ ZMIAN
- **Salience-weighted storage** — zachować, skalować Hebbian outer product przez salience
- **Interference forgetting** — naturalna w SDM (nowe wzorce nakładają się na stare)
- **Consolidation tracking** — zamiast `replay_count` użyć Fisher stiffness (Faza 12)

**Pliki:** `core/episodic_memory.py` (refaktor na SDM), `core/config.py` (SDM params)

**Weryfikacja:**
- 100 wzorców zapisanych, recall z 50% cue → >80% overlap z oryginałem
- Po 10 000 zapisów bez recall: `norm(W_content)` stabilizuje się (nie rośnie liniowo)
- Recall time: O(1) matrix multiply vs O(N) cosine search — benchmark

---

### Faza 12: Konsolidacja Wiedzy — EWC / PKMzeta + Energy Sleep

*Zależności: Faza 11 (SDM jako substrate for Fisher estimation), plan_base.md Phase 7 (cleanup done).*

Cel: Ochrona ugruntowanej wiedzy przed catastrophic forgetting (Kirkpatrick et al. 2017) + termodynamiczne wyzwalanie snu (P2: ciągłość czasowa).

#### 12.1: ConsolidationManager (EWC)

**Nowy plik `core/consolidation.py`:**

`EWCConfig(BaseConfig)`:
- `lambda_ewc: float = 100.0` — siła ochrony
- `tau_stiffness: float = 3_600_000.0` (1 godzina — PKMzeta-like decay, Sacktor 2011)
- `fisher_samples: int = 50` — próbki do estymacji Fishera w sleep

`ConsolidationManager`:
- Per-synapse `stiffness: NDArray[float32]` (diagonala Fishera), init=0

**Podczas online learning (per step):**
$$\Delta w_{ij} = \frac{\text{lr} \times e_{ij}}{1 + \lambda \times \text{stiffness}_{ij}}$$

Wagi z wysoką stiffness opierają się zmianom. Krytyczne umiejętności chronione.

**Podczas sleep SWR (per ripple):**
Fisher estimation z three-factor STDP proxy:
$$F_{ij} \approx \frac{1}{K}\sum_{k=1}^{K} (e_{ij} \times \delta_t)^2$$

NOTA: To jest aproksymacja — SNN nie ma analitycznego ∂L/∂w. Proxy `e_ij × δ_t` jest proporcjonalny do gradientu w three-factor. Lambda może wymagać kalibracji.

**Naturalny rozpad (PKMzeta):**
$$\text{stiffness} \times= \exp(-dt / \tau_{stiffness})$$

Stare umiejętności tracą ochronę na skali godzinowej — pozwala na reorganizację.

**Integracja:** Wstrzykiwany do SNNDeepCritic, D1D2Actor, każdej warstwy z wagami. Modyfikuje `w += lr * dw` → `w += lr * dw / (1 + λ × stiffness)`.

#### 12.2: Energy-Triggered Sleep

**USUNĄĆ z `arena/snn_agent.py`:**
- Blok `if done` wywołujący `replay_buffer.sleep_phase()`

**DODAĆ:**
```
def _check_sleep_need(self) -> bool:
    mean_atp = float(np.mean(self.astrocyte.atp))
    time_awake = (self._step_count - self._last_sleep_step) * self._ctx.dt
    return (mean_atp < sleep_atp_threshold) or (time_awake > max_wake_duration_ms)
```

Wywołanie: na końcu KAŻDEGO `observe()`, nie tylko przy `done`. Agent zasypia kiedy ATP < 30% LUB obudek > 5 minut. Czas trwania snu proporcjonalny do deficytu ATP.

**Dodać do `AgentConfig`:**
- `sleep_atp_threshold: float = 0.3`
- `max_wake_duration_ms: float = 300_000.0`
- `sleep_duration_base_ms: float = 10_000.0`

#### 12.3: Fisher accumulation podczas SWR

Rozszerzyć `replay_buffer._sws_phase()`:
- Po każdym ripple update (krok 7 w istniejącym SWR): obliczyć `grad_proxy = e * td_error` per layer
- `ewc_manager.accumulate_fisher(grad_proxy ** 2)` — KWADRAT (wariancja, nie suma!)
- Po zakończeniu SWS: `ewc_manager.commit_fisher()` — uśrednia i dodaje do stiffness

**Pliki:** nowy `core/consolidation.py`, `core/config.py` (EWCConfig), `arena/snn_agent.py` (energy sleep + EWC integration), `core/replay_buffer.py` (Fisher accumulation w SWR)

**Weryfikacja:**
- Continual learning: ShiftingBandit → TMaze → ShiftingBandit: score po powrocie > 70% oryginału
- Stiffness rośnie na krytycznych wagach podczas sleep
- Stiffness decay: po 1h (simulated) stiffness spada o ~63%
- Energy sleep: 10 000 kroków ciągłej pracy → agent samodzielnie zasypia ≥1 raz

---

### Faza 13: Wiązanie Fazowe — Phase-Gated Propagation

*Zależności: plan_base.md Phase 5 (proper PAC z gamma phase-reset), Faza 9 (axonal delays).*

Cel: Neurony tworzące wspólną reprezentację muszą wypalać w tej samej podfazie gamma. Spike'i "przybywające" w różnych fazach nie integrują się — downstream widzi je jako odrębne obiekty.

#### 13.1: Gamma-phase gated propagation w NetworkGraph

**Modyfikacja `core/network.py` `step()` krok 8 (feedforward aggregation):**

Obecna logika: `agg += connection.weight * output`. Spike'i z każdego źródła sumowane bezwarunkowo.

Nowa logika:
```
gamma_phase = self.oscillator.gamma_phase
phase_window = 2.0 * np.pi * config.coincidence_window_ms * self._ctx.dt * gamma_freq / 1000

for conn in feedforward_connections:
    source_phases = self._layers[conn.source].spike_phase
    arrival_phases = source_phases + conn.delay * 2 * np.pi * gamma_freq / 1000
    phase_diff = np.abs((arrival_phases - gamma_phase) % (2 * np.pi))
    # Koincydencja: arrival phase blisko bieżącej gamma phase
    coincident_mask = (phase_diff < phase_window) | (phase_diff > (2*np.pi - phase_window))
    effective_spikes = source_spikes * coincident_mask
    agg += conn.weight * effective_spikes
```

**Dodać do `AdExLayer`:** `spike_phase: NDArray[float32]` — faza gamma w momencie spike'a. Aktualizowana w `forward()`: `spike_phase[spiked] = current_gamma_phase`.

**Delay correction:** Kluczowe dla Small-World "autostrad". Spike wysłany w fazie φ_send dociera jako φ_arrival = φ_send + delay × ω_gamma. Bez korekcji: 5ms delay przy 40Hz gamma = 72° phase shift → binding się rozpada.

**Dodać do `core/config.py`:** `phase_binding_window_ms: float = 2.0` (okno koincydencji).

UWAGA: To jest UPROSZCZENIE pełnego modelu z plan.md 3.1. Plan.md proponował full arrival-time correction z interneuron-enforced synchronization. Uproszczona wersja implementuje EFEKT (phase-gated propagation) bez MECHANIZMU (interneuron-driven synchronization). Mechanizm jest przyszłym rozszerzeniem.

#### 13.2: Usunięcie concat aggregation

Plan_base.md CLN 1 identyfikuje concat mode jako dead code. Jeśli sum jest jedynym trybem, usunąć concat path z `NetworkGraph._aggregate_inputs()`. Phase-gated propagation zastępuje oba.

**Pliki:** `core/network.py` (phase-gated propagation), `core/neuron.py` (spike_phase tracking), `core/config.py` (phase_binding params)

**Weryfikacja:**
- Dwa sygnały z delay 1ms i 5ms zsynchronizowane po arrival-time correction: downstream odczytuje jako 1 obiekt (korelacja > 0.8)
- Bez phase gating: downstream traktuje jako niezależne (korelacja < 0.3)
- K-WTA nadal działa poprawnie z phase-gated input

---

### Faza 14: Thalamic Gating & Wave Planning

*Zależności: Faza 13 (phase binding musi działać zanim bramkujemy sensory), plan_base.md Phase 6 (HACK C and F removed).*

Cel: Czyste rozdzielenie percepcja/wyobraźnia przez bramkowanie sensoryczne. Planowanie przez dryf atraktorowy (nie for-loop). ACh-gated learning zapobiega halucynacyjnemu uczeniu.

#### 14.1: Thalamic Gate w NetworkGraph

**Dodać do `NetworkGraph`:**
- `thalamic_gate: bool = False`
- W `step()`: gdy `thalamic_gate == True`:
  - `sensory_inputs = {name: np.zeros_like(v) for name, v in sensory_inputs.items()}`
  - Reszta dynamiki bez zmian — identyczne wagi, interneurony, oscylator
  - Sieć "wyobraża sobie" — wewnętrzne atraktory dryfują

#### 14.2: ACh-gated learning (Hasselmo 2006)

**Podczas imagination (thalamic_gate=True):**
- `neuromodulator.set_imagination_mode(True)` → ACh forced to 0.1 (retrieval floor)
- **Efekty na istniejące mechanizmy:**
  - PredictiveCodingLayer: `ach_level=0.1` → dominuje top-down (wewnętrzne)
  - PyramidalLayer: `_ach_apical_scale=0.1` → apical dominant
  - STDP: niska ACh → wolniejsza membrana → mniej spike'ów → mniej STDP
- **BG plasticity NIE jest bramkowana** — DA-driven. Agent MOŻE uczyć wartości z wyobraźni (cel planowania), ale NIE korowe skojarzenia.

#### 14.3: Wave Planning zamiast for-loop mental_rehearsal

**USUNĄĆ z `core/world_model.py`:**
- `EncoderSnapshot` class (save/restore pattern)
- `snapshot_encoder()` / `restore_encoder()`
- `mental_rehearsal()` — cała metoda z iteracją po akcjach

**ZASTĄPIĆ** rozszerzeniem `theta_sweep_plan()` w `core/network.py`:

```
def imagination_plan(self, n_theta_cycles=3):
    """Thalamic gate ON → attractor drift → action readout."""
    self.thalamic_gate = True
    neuromod.set_imagination_mode(True)
    
    theta_period = 1000.0 / self.oscillator.theta_freq
    gamma_period = 1000.0 / self.oscillator.gamma_freq
    n_steps = n_theta_cycles * int(theta_period / gamma_period)
    
    for _ in range(n_steps):
        self.step(sensory_inputs={})  # internal dynamics only
    
    action = actor.get_action()  # which basin did we land in?
    
    self.thalamic_gate = False
    neuromod.set_imagination_mode(False)
    return action
```

System NIE "oblicza" optymalnej akcji — **spływa** do niej po gradiencie EFE. Atraktor z najniższym G(a) ma najgłębszy basen.

#### 14.4: Integracja z ActiveInferenceModule

`ActiveInferenceModule.select_action()` — USUNĄĆ for-loop + softmax. ZASTĄPIĆ: EFE score jako dodatkowy prąd do D1 neurons (plan_base HACK C). Imagination via `imagination_plan()` zamiast `mental_rehearsal()`.

**Pliki:** `core/network.py` (thalamic_gate + imagination_plan), `core/world_model.py` (usunięcie EncoderSnapshot + mental_rehearsal), `core/basal_ganglia.py` (ActiveInferenceModule refaktor), `core/neuromodulator.py` (set_imagination_mode)

**Weryfikacja:**
- Thalamic gate: sensory=zeros, internal dynamics active, ACh < 0.15
- Wagi korowe NIE zmieniają się (STDP gated). Wagi BG ZMIENIAJĄ się (DA-driven).
- Imagination plan: 18 steps → action ≥ 80% zgodna z for-loop baseline
- Brak EncoderSnapshot w codebase: `grep -rn "EncoderSnapshot" core/` → 0

---

### Faza 15: Structural Plasticity (Pruning + Synaptogeneza)

*Zależności: Faza 9 (SmallWorldConnectivity musi istnieć). Faza 12 (EWC chroni ważne wagi przed pruning).*

Cel: Synapse, które nie przenoszą informacji, są usuwane. Nowe synapsy tworzą się w strefach wysokiej Ca²⁺ — aktywności. Dendritic conductance budget (G_max) zapobiega nieskończonemu wzrostowi.

#### 15.1: Pruning martwych synaps

**Warunek usunięcia:**
- `|weight| < ε_prune` (1% initial weight) AND
- `eligibility_trace == 0` przez > `T_prune=5000ms` AND
- `stiffness < stiffness_threshold` (EWC pozwala na usunięcie)

**Implementacja:** Remove from SoA (pre_idx, post_idx, weights, delays). Compact indices periodycznie (co θ cycle).

#### 15.2: Synaptogeneza (nowe synapsy)

**Warunek utworzenia:**
- `Ca²⁺[zone] > synaptogenesis_threshold` (wysoka aktywność) AND
- Pre/post neurons spiked within ±20ms AND
- Brak istniejącego połączenia

**Nowa synapsa:** `weight = 0.1 × σ_init` (mała — musi być wzmocniona przez STDP).
**Limit:** max `k_new_per_cycle` nowych synaps per θ cycle.

#### 15.3: Dendritic Conductance Budget (G_max)

Szybka pętla ujemna zapobiegająca runaway synaptogenesis:
- Każdy neuron: `G_max = k_local × w_init_mean × g_max_factor` (g_max_factor=2.0)
- Przed utworzeniem nowej synapsy: `total_incoming = Σ|w_incoming|`
- Jeśli `total_incoming + w_init > G_max`: competitive displacement — nowa synapsa wypiera najsłabszą

Hierarchia pętli ujemnych:
1. **G_max** — per neuron, per step (natychmiastowa)
2. **Rate limiting** — ±5% zmian topologicznych per θ cycle (średnia)
3. **ATP budget** — per zone, τ~1s (wolna)

**Pliki:** `core/synapse.py` (rozszerzenie SmallWorldConnectivity), `core/config.py` (StructuralPlasticityConfig)

**Weryfikacja:**
- Po 10 000 kroków: ≥5 nowych synaps, ≥3 usunięte
- G_max invariant: nigdy `Σ|w_incoming| > G_max` dla żadnego neuronu
- Connectivity guarantee: min(in_degree) > 0 po plastyczności
- Regresja: ShiftingBandit performance ≥ baseline po structural plasticity enabled

---

### Faza 16: Integration Tests — End-to-End Validation

*Zależności: Wszystkie fazy 9-15.*

#### 16.1: Continual Learning (EWC)
- Agent na sekwencji: ShiftingBandit → TMaze → ShiftingBandit (powrót)
- Score po powrocie > 70% oryginału (EWC chroni)
- Brak explicit task switch z perspektywy agenta

#### 16.2: Skalowalność
- Sieć 5000 neuronów, 60 000 synaps (sparse)
- RAM < 50 MB (vs ~200 MB dense 5K×5K float32)
- Step time < 5ms na CPU (NumPy)

#### 16.3: Natural Sleep Cycle
- 100 000 kroków ciągłej pracy (bez episodów z perspektywy agenta)
- Agent samodzielnie przechodzi cykle wake→sleep→wake (ATP-triggered)
- Performance po śnie > performance tuż przed snem

#### 16.4: Phase Binding
- Dwa kolumny z identical input w same gamma phase → downstream fused
- Dwa kolumny z identical input w different gamma phase → downstream separate
- Verify: inhibitory interneurons enforce synchrony within groups

#### 16.5: VRP Regularization
- Identyczne środowisko, VRP on vs off: VRP agent ma mniejszą wariancję performance (natural regularization)
- STD visible: high-frequency bursts produce diminishing returns

#### 16.6: Anti-Catastrophic-Forgetting
- 500 trials Task A → 500 trials Task B → test Task A
- With EWC: Task A score > 60%
- Without EWC (control): Task A score < 30%

---

## ZALEŻNOŚCI I KOLEJNOŚĆ

```
plan_base.md Phase 2-8 (biophysics cleanup) → PREREQUISITE
    ↓
Faza 9 (Small-World + sparse) ←─── niezależne ───→ Faza 10 (VRP)
    ↓                                                    ↓
Faza 11 (Kanerva SDM) ←── niezależne ──→ Faza 12 (EWC + energy sleep)
    ↓                                          ↓
Faza 13 (Phase Binding) ←── wymaga PAC z plan_base Phase 5
    ↓
Faza 14 (Thalamic Gate + Wave Planning)
    ↓
Faza 15 (Structural Plasticity) ←── wymaga Fazy 9 (sparse) + Fazy 12 (EWC protects weights)
    ↓
Faza 16 (Integration Tests)
```

Parallelizm:
- Fazy 9 + 10 mogą iść równolegle
- Fazy 11 + 12 mogą iść równolegle (po 9/10)
- Faza 13 wymaga plan_base Phase 5 (PAC fix) + Faza 9 (delays)
- Faza 14 wymaga Fazy 13 + plan_base Phase 6
- Faza 15 wymaga Fazy 9 + 12
- Faza 16 po wszystkich

---

## PRYNCYPIA (bez zmian z plan.md)

- **P1: Fizyka zamiast algorytmu** — zachowanie emerguje z dynamiki
- **P2: Ciągłość czasowa** — brak "epizodu" w core
- **P3: Termodynamiczne ograniczenia** — ATP limituje, nie flagi
- **P4: Skalowalność O(N×k)** — sparse topology

## WYKLUCZONE Z ZAKRESU (kolejny plan)

- 1000 Brains / Grid Cells → potrzeba 2D navigation environments
- Homeostatic Drive (Anti-Dark-Room) → potrzeba open-ended environments  
- Hierarchical Delegation → potrzeba multi-subgoal tasks
- Full Variational Inference → predictive coding relaxation wystarczający
- GPU acceleration (CuPy/JAX) → when >10k neurons
- Event-driven simulation → optimization, not correctness
- Cerebellar motor refinement → slot architektoniczny zostawiony
- Multi-agent communication → poza zakresem MVP
