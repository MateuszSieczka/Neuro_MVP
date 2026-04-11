# PLAN TRANSFORMACJI: SNN → Fundament AGI

# Standalone — wystarczający do wykonania wraz z dostępem do kodu

---

## 0. Architektura Obecna (Stan Wyjściowy)

### Struktura Katalogów

- `core/` — 25 modułów Pythona: neuron, synapse, network, config, simulation_context, neuromodulator, oscillator, attention, basal_ganglia, columnar, competitive_layer, episodic_memory, error_neuron, free_energy, interneuron, predictive_coding, pyramidal_neuron, receptor, replay_buffer, sequence_memory, spike_encoder, working_memory, world_model, astrocyte
- `arena/` — 8 modułów: agent_factory, benchmark, core, environments, gym_env, snn_agent, task_config
- Testy w `tests/`, skrypty diagnostyczne: `_diag_mc.py`, `_sweep_all.py`

### Co już działa dobrze

- Dokładna integracja LIF (exponential decay, nie Euler)
- 4-kanałowa neuromodulacja (DA, ACh, NE, 5-HT) z farmakokinetykami i równaniem Hilla
- Kanały synaptyczne AMPA/NMDA/GABA-A/GABA-B z blokiem magnezowym Jahra-Stevensa
- D1/D2 Actor z dwustanem MSN (Up/Down) i oddzielnymi ścieżkami Go/NoGo
- Oscylator theta-gamma z PAC (Phase-Amplitude Coupling)
- Predictive Coding z modulacją ACh (Rao & Ballard 1999)
- Neurony piramidalne z przedziałami apikalnymi i Ca²⁺ spike (Larkum 2013)
- Astrocyty z dynamiką Ca²⁺, D-Seryną, dyfuzją gap-junction
- Three-factor STDP (eligibility × modulator × error)
- Prawo Dale'a (neurony albo E albo I)

### Co wymaga fundamentalnej zmiany

- **Model neuronu LIF** — brak adaptacji, nie generuje intrinsic bursts
- **Gęste macierze wag** — O(N²) pamięci, nie skaluje się
- **Episodyczność** — `is_terminal`, `episode_steps`, tonic DA per episode
- **Softmax + np.random.choice** — w D1D2Actor.forward() i ActiveInferenceModule.select_action()
- **Concat agreagacja** — `aggregation_mode="concat"` w NetworkGraph
- **EncoderSnapshot** — kopiowanie stanu RAM do wyobraźni
- **For-loop planowanie** — `for action in candidate_actions` w mental_rehearsal()
- **Lista pamięci O(N)** — cosine similarity search w EpisodicMemory
- **~30 magic numbers** — hardkodowane wartości w snn_agent.py bez derywacji
- **Brak ochrony wiedzy** — catastrophic forgetting przy continuous learning
- **Brak budżetu energetycznego** — sieć może wypalać w nieskończoność

---

## 1. Pryncypia Architektury Docelowej

Każda decyzja implementacyjna musi przejść test czterech pryncypiów:

**P1. Fizyka zamiast algorytmu.** Żadnego obliczania rozkładów prawdopodobieństwa (softmax), żadnych pętli przeszukujących kandydatów (`for action in`), żadnych instrukcji warunkowych typu `if energy < threshold: stop`. Zachowanie emerguje z dynamiki równań różniczkowych.

**P2. Ciągłość czasowa.** Brak pojęcia "epizod" czy "iteracja" w rdzeniu systemu. Agent funkcjonuje w ciągłym strumieniu czasu. Sen wynika z termodynamiki (wyczerpanie ATP), nie z flagi `done`.

**P3. Termodynamiczne ograniczenia.** Każde wyładowanie kosztuje energię (ATP). Brak energii fizycznie podnosi próg pobudliwości i spowalnia dynamikę — nie blokuje wyładowania instrukcją warunkową.

**P4. Skalowalność O(N×k).** Żadnych gęstych macierzy N×N. Topologia Small-World z plastycznością strukturalną.

---

## FAZA 0: Higiena Kodu

**Cel:** Uzdrowienie bazy kodu bez zmian behawioralnych. Warunek konieczny — każda późniejsza faza buduje na czystym fundamencie.

### Krok 0.1: Ekstrakcja magic numbers do configów

**Pliki:** `arena/snn_agent.py`, `core/config.py`

Utworzyć `AgentConfig(BaseConfig)` w `core/config.py` z polami:

- `intrinsic_reward_weight: float = 0.1` (waga curiosity w effective_reward)
- `da_offset: float = 0.5` (przesunięcie learning_rate_modulation → DA level)
- `td_clip: float = 50.0` (gradient clipping na TD error)
- `consolidation_midpoint: float = 0.7` (punkt przegięcia sigmoidy konsolidacji)
- `consolidation_steepness: float = 8.0` (stromość sigmoidy)
- `consolidation_floor: float = 0.8` (minimum plasticity scale)
- `noise_smoothing: float = 0.8` (EMA exploration noise)
- `min_exploration: float = 0.15` (podłoga eksploracji)
- `sleep_gain_scale: float = 0.5` (mnożnik quality→sleep_gain)
- `sleep_gain_max: float = 2.5`

W `arena/agent_factory.py` — przenieść do parametrów fabryki:

- `columnar_threshold: int = 16`
- `default_receptive_field: int = 4`

W `arena/benchmark.py`:

- `default_seeds: list[int]` do `BenchmarkConfig`
- `solve_rate_threshold: float = 0.5`

W `arena/task_config.py`:

- Dodać brakujące pole `reward_scale: float = 1.0` do `TaskConfig` — `benchmark.py` przekazuje `task.reward_scale` do `GymEnv`, ale pole nie istnieje w dataclass. Runtime `AttributeError`.

### Krok 0.2: Usunięcie dostępu do prywatnych atrybutów

W `core/neuromodulator.py` — dodać publiczne property:

```
@property
def reward_history(self) -> list[float]
```

zamiast `self.neuromod._reward_history` w `snn_agent.py`.

W `arena/snn_agent.py` — property `use_world_model: bool` (publiczne) zamiast `_use_wm`.

**UWAGA:** `self.neuromod._reward_history` w snn_agent NIE ISTNIEJE — `NeuromodulatorSystem` nie posiada atrybutu `_reward_history`. Guard `hasattr()` zawsze zwraca `False`, więc `sleep_gain = 1.0` zawsze. Trzeba dodać `_reward_history: deque[float]` do `NeuromodulatorSystem.__init__()` i populować w `update_tonic_da()`, a następnie wyeksponować przez property.

### Krok 0.3: Unifikacja logiki homeostazy

Trzy warstwy (`CompetitiveLIFLayer`, `PyramidalLayer`, `ErrorNeuronLayer`) niezależnie zarządzają `v_thresh_adaptive` i `avg_rate`. Wyekstrahować stan homeostatyczny:

```
HomeostaticState:
  v_thresh_adaptive: NDArray
  avg_rate: NDArray
  is_dark_matter: NDArray

  update(spikes, config) → None
  effective_threshold(ne_level, config) → NDArray
```

Wszystkie trzy warstwy używają tego samego obiektu. Eliminuje duplikację ~60 linii kodu.

### Krok 0.4: Walidacja configów

W `__post_init__` każdego configa dodać asserty:

- `NeuronConfig`: `assert tau_m > 0`, `assert v_reset < v_thresh`, `assert v_rest < v_thresh`
- `SynapseConfig`: `assert tau_ampa < tau_nmda`, `assert mg_concentration > 0`
- `STDPConfig`: `assert tau_plus > 0`, `assert tau_minus > 0`, `assert a_plus > 0`
- `HomeostaticConfig`: `assert 0 < target_rate < 1`, `assert thresh_min < thresh_max`
- Itd. dla każdego configa — pattern: `assert invariant, "message"`

### Krok 0.5: Naprawić pyproject.toml

```toml
packages = [{include = "core"}, {include = "arena"}]
requires-python = ">=3.11"

[tool.poetry.group.dev.dependencies]
pytest = ">=7.0"
```

### Krok 0.6: Naprawić bias Welford w gym_env.py

Obecny kod: `self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count`
Poprawić na: dzielenie przez `self._obs_count - 1` gdy `self._obs_count >= 2` (nieobciążony estymator Bessela).

### Krok 0.7: Naprawić bug w PyramidalLayer.generate_prediction()

**Plik:** `core/pyramidal_neuron.py`, `core/config.py`

`generate_prediction()` odwołuje się do `self.pc_cfg.feedback_strength`, ale `PyramidalLayer` dziedziczy z `CompetitiveLIFLayer`, NIE z `PredictiveCodingLayer` — atrybut `pc_cfg` nie istnieje. Runtime `AttributeError` przy wywołaniu.

Naprawić:

1. Dodać `feedback_strength: float = 0.5` do `PyramidalConfig`
2. Zmienić `self.pc_cfg.feedback_strength` → `self.pyr_cfg.feedback_strength`

### Weryfikacja Fazy 0

- `python -m pytest tests/` — wszystkie testy przechodzą bez zmian
- `grep -rn "self\.\(neuromod\|agent\)\._" arena/` — zero wyników na prywatne atrybuty z zewnątrz
- Każdy config akceptuje poprawne wartości, rzuca `AssertionError` na niepoprawne
- `PyramidalLayer().generate_prediction()` nie rzuca `AttributeError`
- `TaskConfig` posiada pole `reward_scale` — `benchmark.py` nie rzuca `AttributeError`

**NOTA o testach:** Każda kolejna faza (1-4) zmienia publiczne API modułów. Testy będą wymagały aktualizacji po każdej fazie, zachowując tę samą strukturę i cel. Faza 0 jest wyjątkiem — NIE zmienia zachowania, więc testy MUSZĄ przejść bez modyfikacji.

---

## FAZA 1: Biofizyka Czasu Ciągłego

**Cel:** Zastąpienie modelu LIF modelem AdEx, wprowadzenie stabilnego integratora numerycznego, budżetu energetycznego, i ciągłej dynamiki dopaminy.

**Zależności:** Wymaga ukończenia Fazy 0.

### Krok 1.1: Model AdEx (Brette & Gerstner 2005)

**Pliki:** `core/config.py`, `core/neuron.py`

Dodać do `NeuronConfig` nowe pola:

- `delta_t: float = 2.0` — ostrość inicjacji spike'a (mV). Przy V bliskim V_T, prąd eksponencjalny narasta
- `v_spike_cutoff: float = -30.0` — próg detekcji spike'a (mV), wyższy niż V_T
- `tau_w: float = 144.0` — stała czasowa prądu adaptacyjnego (ms)
- `a: float = 4.0` — konduktancja adaptacji podprogowej (nS)
- `b: float = 80.5` — inkrement adaptacji po spike'u (pA)
- `g_L: float = 30.0` — konduktancja upływu (nS, Destexhe & Paré 1999)
- `C_m: float = 281.0` — pojemność membrany (pF)

Pochodne w `__post_init__`:

- `w_decay: float = exp(-dt/tau_w)`
- `w_gain: float = 1 - exp(-dt/tau_w)`

Równanie membranowe AdEx:
$$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T \exp\!\left(\frac{V - V_T}{\Delta_T}\right) + I_{syn} - w$$

Równanie adaptacji:
$$\tau_w \frac{dw}{dt} = a(V - E_L) - w$$

Po spike'u ($V \geq V_{cutoff}$):
$$V \leftarrow V_{reset}, \quad w \leftarrow w + b$$

W `core/neuron.py` — zmienić `LIFLayer` na `AdExLayer`:

- Dodać stan `w_adapt: NDArray[float32]` (prąd adaptacyjny per neuron)
- W `forward()`: zamienić liniową integrację na Exponential Euler z członem AdEx (Krok 1.2)
- Detekcja spike'a: `v >= v_spike_cutoff` zamiast `v >= v_thresh`
- Po spike'u: `w_adapt += b`
- Adaptacja: `w_adapt = w_adapt * w_decay + a * (v - v_rest) * w_gain`

Różne typy neuronów z TYCH SAMYCH równań — różnią się tylko parametrami:

- **Regular Spiking (kora, piramidalne):** a=4, b=80.5, tau_w=144
- **Fast Spiking (interneurony PV+):** a=0, b=0, tau_w=144, delta_t=2 → brak adaptacji
- **Intrinsic Bursting (L5 piramidalne):** a=2, b=60, tau_w=20 → szybka adaptacja → burst
- **Late Spiking (SOM+):** a=-2, b=0 → opóźniony firing

Wszystkie warstwy potomne (`CompetitiveLIFLayer`→`AdExCompetitive`, `PredictiveCodingLayer`, `PyramidalLayer`, `ErrorNeuronLayer`, `InhibitoryPool`, `SNNDeepCritic`, `D1D2Actor`) dziedziczą zmianę automatycznie — operują na v, w_adapt zamiast samego v.

**UWAGA implementacyjna:** "Automatycznie" jest uproszczeniem. Następujące warstwy NADPISUJĄ `forward()` z własną logiką membranową i wymagają ręcznej adaptacji do AdEx:

- `CompetitiveLIFLayer.forward()` — wywołuje `super().forward()` → automatyczne
- `PredictiveCodingLayer.forward()` — własna integracja → ręczna zmiana
- `PyramidalLayer.forward()` — własna integracja z apical → ręczna zmiana
- `ErrorNeuronLayer.forward()` — dwie oddzielne populacje (state/error) → ręczna zmiana
- `InhibitoryPool.forward()` — własna LIF integracja → ręczna zmiana
- `SNNDeepCritic._forward_pop()` — własna LIF integracja → ręczna zmiana
- `D1D2Actor.forward()` — bistable MSN dynamics → ręczna zmiana

### Krok 1.2: Integrator Exponential Euler

**Plik:** `core/simulation_context.py`

Obecna metoda `decay(tau)` jest analitycznie dokładna dla liniowego LIF. Człon eksponencjalny AdEx ($g_L \Delta_T \exp((V-V_T)/\Delta_T)$) i blok magnezowy NMDA ($B(V) = 1/(1 + [Mg]/3.57 \cdot e^{-0.062V})$) to nieliniowości wymagające stabilnego schematu.

Dodać metodę do `SimulationContext`:

```
exp_euler_step(v, F_v, J_v, dt) → v_next
```

Gdzie:

- $F(v)$ — pełna prawa strona ODE: $F(V) = \frac{1}{C_m}[-g_L(V-E_L) + g_L\Delta_T e^{(V-V_T)/\Delta_T} + I - w]$
- $J(v) = \partial F / \partial V$ — Jakobian (skalar per neuron)
- Schemat Exponential Rosenbrock 1. rzędu:
  $$V_{n+1} = V_n + \varphi_1(h \cdot J) \cdot h \cdot F(V_n)$$
  $$\varphi_1(z) = \frac{e^z - 1}{z}$$

**Implementacja φ₁ z pełną precyzją float32:**

Bezpośrednie `(exp(z) - 1) / z` traci ~7 bitów mantysy przy z ≈ 10⁻³ (catastrophic cancellation w odejmowaniu bliskich wartości). Użyć sprzętowo zoptymalizowanego `np.expm1()`:

```
phi1 = np.where(np.abs(z) < 1e-4, 1.0 + z / 2.0, np.expm1(z) / z)
```

`np.expm1(z)` oblicza `eᶻ - 1` jako JEDNĄ operację z pełną precyzją mantysową. Branch `1 + z/2` dla |z| < 10⁻⁴ to Taylor drugiego rzędu — wystarczający bo |error| < z²/6 ≈ 10⁻⁹.

Dlaczego nie RK4: stiffness NMDA (τ=100ms vs AMPA τ=2ms = ratio 50:1) wymaga A-stabilnego integratora. Exponential Euler jest A-stabilny z kosztem O(N), nie O(N³) jak implicit.

### Krok 1.3: Budżet Energetyczny ATP (ciągły, nie binarny)

**Plik:** `core/astrocyte.py`, `core/config.py`

Dodać do `AstrocyteConfig`:

- `atp_max: float = 1.0` (znormalizowany pułap)
- `atp_regen_rate: float = 0.001` (regeneracja per ms — ~1s do pełni)
- `atp_spike_cost: float = 0.02` (koszt per spike per zona)
- `atp_threshold_shift: float = 10.0` (mV — maks. przesunięcie progu przy zerowym ATP)
- `atp_leak_gain: float = 0.5` (maks. wzrost konduktancji upływu przy zerowym ATP)

W `AstrocyteField` dodać stan:

- `atp: NDArray[float32]` per zona, zainicjalizowane na `atp_max`

Aktualizacja w `update()`:

```
# Regeneracja (first-order recovery z saturacją)
atp += atp_regen_rate * (atp_max - atp) * dt

# Koszt aktywności (proporcjonalny do spike count per zone per step)
atp -= atp_spike_cost * zone_spike_counts * dt
atp = maximum(atp, 0)
```

**KRYTYCZNE: ATP nie bramkuje wyładowania binarnie.** Zamiast tego eksportuje ciągłe modulacje:

```
@property
def threshold_shift(self) -> NDArray[float32]:
    """Przesunięcie progu V_T w górę przy niskim ATP (mV).

    Biologia: spadek ATP → Na⁺/K⁺-ATPase zwalnia → gradienty jonowe
    słabną → próg inicjacji spike'a rośnie (Na⁺ kanały potrzebują
    większej depolaryzacji). Efekt ciągły.
    """
    return atp_threshold_shift * (1 - atp)  # 0 mV przy pełnym ATP, +10 mV przy zerowym

@property
def leak_gain(self) -> NDArray[float32]:
    """Mnożnik konduktancji upływu g_L.

    Biologia: spadek ATP → pompa Na/K zwalnia → potencjał spoczynkowy
    depolaryzuje → ale gradient K+ słabnie → netto: szybszy rozpad
    potencjału → mniejsza integracja wejść.
    """
    return 1.0 + atp_leak_gain * (1 - atp)  # 1.0 przy pełnym, 1.5 przy zerowym
```

W `AdExLayer.forward()` — stosować ciągłe modulacje:

```
effective_V_T = V_T + astrocyte.threshold_shift[zone_idx]
effective_g_L = g_L * astrocyte.leak_gain[zone_idx]
```

Efekt: sieć płynnie cichnie pod wyczerpaniem — brak instrukcji `if`. Naturalna rzadkość emergentna z termodynamiki.

### Krok 1.4: Dopamina jako Ciągły Leaky Integrator

**Plik:** `core/neuromodulator.py`

USUNĄĆ:

- Metodę `update_tonic_da(episode_return, episode_steps, pred_error_avg)` — episodyczna
- Zmienne: `_welford_n`, `_welford_mean`, `_welford_m2` — statystyki episodyczne
- Zmienne: `_episode_pred_errors` — bufor per-episode
- Zmienną: `_smoothed_reward` — wygładzanie episodowego sygnału

DODAĆ do `NeuromodulatorConfig`:

- `tau_tonic_da: float = 60_000.0` (60 sekund — minutowa skala, Grace 1991)

ZMIENIĆ w `update()` (per-step, nie per-episode):

```
# Tonic DA: ciągła całka sygnału RPE (leaky integrator)
rpe_abs = abs(td_error)
tonic_decay = ctx.decay(tau_tonic_da)
tonic_da = tonic_da * tonic_decay + rpe_abs * (1 - tonic_decay)
```

Tonic DA teraz ewaluuje performance w oknie ~minutowym bez pojęcia "epizodu".

### Krok 1.5: Usunięcie Granic Epizodów z Core

**Plik:** `arena/snn_agent.py`

USUNĄĆ z `observe()`:

- Blok `if done:` wywołujący `neuromod.update_tonic_da()`
- Zmienne `_episode_return`, `_episode_steps`
- Referencja do `self.neuromod._reward_history` (Krok 0.2 daje publiczne API)

ZACHOWAĆ w `arena/core.py`:

- `done` flag w `Trainer._run_episode()` — środowisko nadal ma naturalne końce
- `agent.reset()` — resetuje napięcia membranowe, NIE wagi ani wiedzę

Agent nie rozróżnia "koniec gry" od "kolejna chwila". To Trainer resetuje środowisko — agent widzi ciągły strumień (state, reward, next_state).

### Weryfikacja Fazy 1

1. **AdEx patterns:** Test reprodukujący Brette & Gerstner 2005 Fig. 2: Regular Spiking (ISI rośnie), Bursting (3-5 spike'ów na burst), Fast Spiking (stały ISI). Identyczne parametry z paper.
2. **Stabilność numeryczna:** 10 000 kroków z NMDA + AdEx, brak NaN, brak v → ∞.
3. **ATP ciągłość:** Przy stałej stymulacji: v_thresh_effective rośnie monotonicznie, firing rate maleje asymptotycznie, nigdy nie spada skokowo do zera.
4. **Tonic DA:** Pod stałym reward → tonic_da stabilizuje się; pod reward switch z +1 na -1 → tonic_da relaksuje do nowej wartości w ~60s.

---

## FAZA 2: Topologia, Skalowalność i Pamięć Atraktorowa

**Cel:** Złamanie O(N²), ochrona wiedzy, biologiczna stochastyka synaptyczna.

**Zależności:** Wymaga ukończonej Fazy 1 (AdEx musi być stabilny zanim zmienimy topologię).

### Krok 2.1: Rzadka Konektywność Small-World (Watts-Strogatz)

**Pliki:** `core/config.py`, `core/synapse.py` (rozszerzenie), `core/neuron.py`

Dodać `TopologyConfig(BaseConfig)`:

- `k_local: int = 6` — lokalne połączenia (per stronę, łącznie 2k)
- `p_rewire: float = 0.1` — prawdopodobieństwo przeokablowania (Watts & Strogatz 1998)
- `use_sparse: bool = True`

Dodać klasę `SmallWorldConnectivity` w `core/synapse.py`:

- `__init__(n_pre, n_post, config)` → generuje wzorzec połączeń Watts-Strogatz
- Przechowuje jako SoA (Structure of Arrays): `pre_idx, post_idx, weights, delays` — cztery wektory 1D NumPy
- `multiply(pre_spikes) → post_current`:

  **UWAGA:** Naiwne `current[post_idx] += weighted` jest BŁĘDNE — NumPy fancy indexing z duplikatami liczy tylko ostatni zapis, nie sumuje. Dwie opcje:
  - `np.add.at(current, post_idx, pre_spikes[pre_idx] * weights)` — poprawne, unbuffered, ale nie wektoryzowane przez SIMD
  - `np.bincount(post_idx, weights=pre_spikes[pre_idx] * weights, minlength=n_post)` — w pełni wektoryzowane, ~3-5× szybsze na dużych sieciach. **Preferowana opcja.**

- `update_weights(delta_w_sparse)` — aktualizacja in-place na wektorze weights
- Pamięć: O(n_pre × k_local) zamiast O(n_pre × n_post)

W `core/neuron.py` `AdExLayer`:

- Jeśli `topology_cfg` podane: `self.connectivity = SmallWorldConnectivity(...)` zamiast gęstego `self.w`
- `forward()`: `current = self.connectivity.multiply(pre_spikes)` zamiast `pre_spikes @ w`
- STDP: operuje na `connectivity.weights` (sparse), nie na pełnej macierzy

**Opóźnienia aksonalne per-synapsa:**

Zamiast `(pre_idx, post_idx, weights)` struktura sparse przechowuje `(pre_idx, post_idx, weights, delays)`.

Dodać do `TopologyConfig`:

- `local_delay_ms: float = 1.0` (propagacja lokalna w ringu)
- `long_range_delay_factor: float = 0.5` (ms per hop oryginalnej odległości w ringu)

Delays wyznaczane przy tworzeniu krawędzi:

- Lokalne krawędzie (ring): `delay = local_delay_ms`
- Rewired "autostrady": `delay = d_ring × long_range_delay_factor` gdzie `d_ring` = odległość w oryginalnym ringu przed rewiring

`multiply()` używa delay bufferów — grupuje synapsy po delay, batch scatter-gather per unique delay value. Wzorzec do replikacji: `output_history` deque w istniejącym `NetworkGraph`.

**Gwarancja spójności grafu:**

Po inicjalizacji Watts-Strogatz ORAZ po każdym cyklu structural plasticity (Krok 2.2):

```
assert min(in_degree) > 0 and min(out_degree) > 0
# Jeśli neuron osierocony → dodaj krawędź do najbliższego sąsiada w ringu (fallback)
```

Ryzyko disconnection jest niskie (k_local=6 → degree=12, P(all edges rewired away) ≈ 10⁻⁶), ale structural pruning MOŻE osierocić neurony → gwarancja jest konieczna.

Clustering coefficient C ≈ 3(k-1)/2(2k-1) ≈ 0.5 (gęste lokalne klastry).
Average path length L ≈ N/(2k) × ln(N) (logarytmiczne — "autostrady" zapewniają globalną synchronizację).

### Krok 2.2: Plastyczność Strukturalna (Synaptogeneza i Pruning)

**Plik:** `core/synapse.py`

Dodać klasę `StructuralPlasticity`:

**Synaptogeneza** (tworzenie nowych synaps):

- Warunek: `Ca²⁺[zone] > synaptogenesis_threshold` AND `pre_neuron spiked within ±20ms of post_neuron` AND brak istniejącego połączenia
- Nowa synapsa: `weight = 0.1 × σ_init` (mała, musi być wzmocniona przez STDP żeby przeżyć)
- Limit: max `k_new_per_cycle` nowych synaps per cykl theta
- Implementacja: append do sparse vectors (pre_idx, post_idx, weights)

**Pruning** (usuwanie martwych synaps):

- Warunek: `|weight| < ε_prune` AND `eligibility_trace == 0` przez > `T_prune` kroków
- `ε_prune = 0.01 × σ_init` (1% wagi inicjalnej)
- `T_prune = 5000ms` (5 sekund — jeśli synapsa nie bierze udziału w STDP przez 5s, jest martwa)
- Implementacja: remove from sparse vectors, compact indices

Rate limiting zapobiega niestabilności: max ±5% zmian topologicznych per cykl theta.

**KRYTYCZNE: Dendritic Conductance Budget (G_max) — szybka pętla ujemna:**

Rate limiting (±5%/theta) spowalnia crescendo, ale NIE przerywa pętli dodatniego sprzężenia: nowa synapsa → więcej spike'ów → więcej Ca²⁺ → więcej synaptogenezy → ... . ATP jest hamulcem WOLNYM (τ~1s). Potrzebna jest SZYBKA lokalna pętla ujemna.

Dodać do `TopologyConfig`:

- `g_max_factor: float = 2.0` (max konduktancja = g_max_factor × initial_total_conductance)

Każdy neuron ma `G_max = k_local × w_init_mean × g_max_factor`. Przed utworzeniem nowej synapsy:

```
total_incoming = sum(|weights[post_idx == neuron_id]|)
if total_incoming + w_init > G_max:
    # Competitive displacement: nowa synapsa wypiera najsłabszą istniejącą
    weakest_idx = argmin(|weights[post_idx == neuron_id]|)
    if |weights[weakest_idx]| < w_init:
        replace(weakest_idx, new_synapse)
    else:
        reject(new_synapse)  # istniejące synapsy silniejsze → nowa przegrywa
```

Biologicznie: dendritic spine competition (Holtmaat & Svoboda 2009) — nowe kolce rywalizują z istniejącymi o ograniczone scaffolding proteins i powierzchnię dendrytu. Neuron FIZYCZNIE nie może mieć nieskończenie wielu synaps.

Hiererchia pętli ujemnych:

1. **G_max** — per neuron, per step (natychmiastowa)
2. **Rate limiting ±5%/theta** — per zone, per ~167ms (średnia)
3. **ATP budget** — per zone, τ~1s (wolna)

### Krok 2.3: Kanerva SDM jako Pamięć Epizodyczna

**Plik:** `core/episodic_memory.py`

USUNĄĆ:

- Listę Pythona `self._memories: list[Episode]`
- Pętlę cosine similarity search w `recall()`
- `mark_replayed()` counter

ZASTĄPIĆ Kanerva Sparse Distributed Memory:

**Architektura:**

1. **Address Decoder (= istniejący Dentate Gyrus):**
   - Wejście: state vector (N dim)
   - DG sparse expansion: N → M (M = expansion_factor × N, np. 5×)
   - k-WTA: top 5% aktywnych = sparse binary address `a ∈ {0,1}^M`
   - To JUŻ istnieje w kodzie (`dg_sparsity=0.05`, `dg_expansion_factor=5`)

2. **Content Matrix (= CA3):**
   - `W_content: NDArray[float32]` shape (M, content_dim) — DG address → stored content
   - Zapis: `W_content += a[:, newaxis] × content[newaxis, :]` (outer product, Hebbian)
   - Odczyt: `recalled = a @ W_content → content_dim vector`
   - Normalizacja: `recalled /= max(sum(a), 1)` (average over active addresses)

3. **Pojemność:**
   - SDM z M adresów i sparsity p = 0.05: pojemność ∝ exp(M × H(p)) ≈ eksponencjalna w M
   - Dla M=320 (64 × 5): pojemność ~10⁴ wzorców (vs. 500 fixed capacity w obecnym kodzie)

4. **Recall = O(1):**
   - `a = DG_encode(cue)` — sparse projection
   - `content = a @ W_content / sum(a)` — matrix-vector multiply
   - Brak iteracji, brak przeszukiwania listy

5. **NE-gated storage:** Zachować istniejący mechanizm — storage tylko gdy `NE > ne_threshold`

6. **Interference forgetting:** Naturalna w SDM — nowe wzorce nakładają się na stare w W_content. Częściej odtwarzane wzorce mają silniejsze ślady (Hebbian consolidation).

7. **Decay zapobiegający nieograniczonemu wzrostowi W_content:**
   - Per step: `W_content *= (1 - sdm_decay_rate)` z `sdm_decay_rate: float = 1e-5` (w config)
   - Po zapisie: normalizacja aktywnych rzędów: `W_content[active_rows] /= max(norm(W_content[active_rows], axis=1, keepdims=True), 1.0)`
   - Bez decay: akumulacja `W += a⊗content` rośnie bez ograniczeń → degradacja recall, potencjalna niestabilność numeryczna
   - Biologicznie: synapsy CA3 podlegają turnover — stare ślady blakną bez rekonsolidacji

### Krok 2.4: EWC / PKMzeta (Ochrona Ugruntowanej Wiedzy)

**Plik:** nowy `core/consolidation.py`, `core/config.py`

Dodać `EWCConfig(BaseConfig)`:

- `lambda_ewc: float = 100.0` (siła ochrony — wyższa = sztywniejsze ważne wagi)
- `tau_stiffness: float = 3_600_000.0` (rozpad stiffness — 1 godzina, PKMzeta-like)
- `fisher_samples: int = 50` (próbki do estymacji Fishera w sleep)

Dodać klasę `ConsolidationManager`:

- Per-synapse `stiffness: NDArray[float32]` (diagonala informacji Fishera), zainicjalizowana na 0
- **Podczas online learning (per step):**
  $$\Delta w_{ij} = \frac{\text{lr} \times e_{ij}}{1 + \lambda \times \text{stiffness}_{ij}}$$
  Wagi z wysoką stiffness opierają się zmianom — krytyczne umiejętności chronione.
- **Podczas sleep consolidation (per SWR burst):**
  Estymacja diagonali Fishera z gradientów replayowanych doświadczeń:
  $$F_{ij} \approx \frac{1}{K}\sum_{k=1}^{K} \left(\frac{\partial \mathcal{L}_k}{\partial w_{ij}}\right)^2$$
  W naszym kontekście: $\frac{\partial \mathcal{L}}{\partial w} \propto e_{ij} \times \delta_t$ (eligibility × TD error)
  **NOTA:** To jest APROKSYMACJA — SNN nie posiada analitycznego gradientu ∂L/∂w. Proxy `e_ij × δ_t` jest proporcjonalny do gradientu w three-factor STDP, ale nie jest ścisłą diagonalą Fishera. Wystarczająca do regularyzacji, ale parametr `lambda_ewc` może wymagać dostrojenia empirycznego.
  $$\text{stiffness}_{ij} += F_{ij}$$

- **Naturalny rozpad (PKMzeta model):**
  $$\text{stiffness} \times= \exp(-dt / \tau_{stiffness})$$
  Stare umiejętności stopniowo tracą ochronę (τ ~ godzina), pozwalając na reorganizację.

Integracja: `ConsolidationManager` jest wstrzykiwany do każdej warstwy z wagami. Modyfikuje krok `w += lr * dw` na `w += lr * dw / (1 + λ * stiffness)`.

### Krok 2.5: Dynamika Uwolnienia Pęcherzyków (VRP)

**Plik:** `core/synapse.py` (rozszerzenie `SynapticChannels`), `core/config.py`

Dodać `VRPConfig(BaseConfig)`:

- `n_docked_max: int = 10` (max pęcherzyków w docked pool)
- `p_release_base: float = 0.3` (bazowe prawdopodobieństwo uwolnienia — nie 0.8!)
- `tau_recovery: float = 800.0` (ms — recovery docked vesicles, Zucker & Regehr 2002)
- `tau_facilitation: float = 200.0` (ms — decay facilitation)
- `facilitation_increment: float = 0.05` (per spike increase in p_release)

Stan per-synapsa (wektoryzowany):

- `n_docked: NDArray[float32]` (pęcherzyki gotowe do uwolnienia)
- `p_release: NDArray[float32]` (bieżące prawdopodobieństwo)

Na pre-spike:

```
released = Binomial(n_docked, p_release)  # stochastyczne uwolnienie
n_docked -= released
effective_weight = weight * released / n_docked_max  # skalowanie siły synapsy
```

Między spike'ami:

```
# Recovery (docked pool się uzupełnia)
n_docked += (n_docked_max - n_docked) * (1 - exp(-dt/tau_recovery))

# Facilitation decays (p_release wraca do baseline)
p_release = p_release * exp(-dt/tau_facilitation) + p_release_base * (1 - exp(-dt/tau_facilitation))
```

Na pre-spike (facilitacja):

```
p_release += facilitation_increment * (1 - p_release)  # zwiększa p, saturuje przy 1
```

Efekty:

- **Short-term depression (STD):** częste spike'i → n_docked maleje → mniej uwolnienia → synapsa słabnie czasowo
- **Short-term facilitation (STF):** seria spike'ów → p_release rośnie → następne spike'i silniejsze
- **Naturalny dropout:** stochastyczność Binomial(n,p) → regularyzacja bez sztucznego P_r=0.8

USUNĄĆ z `core/world_model.py`: stałe `_vesicle_p = 0.8` i `_decode_with_vesicle_noise()` z Bernoulli mask. VRP daje ten sam efekt fizycznie.

### Weryfikacja Fazy 2

1. **Small-World:** Graf 1000 neuronów, k=6, p=0.1: clustering C > 0.3, avg path < 10 (vs. C ≈ 0 i L ≈ 500 dla random Erdős–Rényi z tym samym k)
2. **Connectivity guarantee:** Po init i po 1000 krokach structural plasticity: `min(in_degree) > 0`, `min(out_degree) > 0`
3. **Axonal delays:** Long-range (rewired) edges mają delay > local edges. `max(delays) < gamma_period` (inaczej spike "przeskakuje" cały cykl)
4. **Pamięć SDM:** Zapisać 100 wzorców, recall z 50% cue → >80% overlap z oryginałem
5. **SDM decay:** Po 10 000 zapisów bez recall: `norm(W_content)` stabilizuje się (nie rośnie liniowo)
6. **EWC:** Uczyć Task A (CartPole) → sleep → uczyć Task B (MountainCar) → test Task A: score > 80% oryginalnego (uwaga: na ten moment cartpole nie działa)
7. **VRP STD:** Pre spike train 100Hz przez 500ms → effective weight spada do <30% po 200ms, recovery w ~1s
8. **Structural plasticity:** Po 10 000 kroków: ≥ 5 nowych synaps powstało, ≥ 3 martwe usunięte
9. **G_max enforcement:** Nigdy `sum(|w_incoming|) > G_max` dla żadnego neuronu (invariant test)
10. **Regresja pamięci:** Test na O(N×k) pamięci zamiast O(N²)

---

## FAZA 3: Architektura Kognitywna i Kompozycyjność

**Cel:** Eliminacja softmax/concat, wdrożenie wiązania przez synchronię, autonomicznych kolumn, wrodzonych priorów homeostatycznych.

**Zależności:** Wymaga Fazy 2 (sparse connectivity potrzebna do precyzyjnego timing'u fazowego).

### Krok 3.1: Wiązanie przez Synchronię Fazową (Binding by Synchrony)

**Pliki:** `core/network.py`, `core/columnar.py`, `core/config.py`

USUNĄĆ z `core/network.py`:

- `aggregation_mode="concat"` z `LayerConnection`
- `np.concatenate()` w `_aggregate_inputs()`

ZASTĄPIĆ mechanizmem detekcji koincydencji fazowej:

Każdy neuron (w AdExLayer) dostaje dodatkowy stan:

- `spike_phase: NDArray[float32]` — faza gamma w momencie ostatniego spike'a

Detekcja wiązania w `NetworkGraph.step()`:

```
# Po propagacji wszystkich warstw:
phase_window = 2.0  # ms — okno koincydencji

for target in downstream_layers:
    incoming_spikes = []
    incoming_phases = []
    incoming_delays = []
    for source in source_layers:
        spikes = outputs[source]
        phases = layers[source].spike_phase
        delays = connectivity[source→target].delays  # per-connection axonal delay (Krok 2.1)
        incoming_spikes.append(spikes)
        incoming_phases.append(phases)
        incoming_delays.append(delays)

    # KRYTYCZNE: Koincydencja używa ARRIVAL TIME, nie SEND TIME
    # Spike wysłany w fazie φ_send dociera w fazie φ_arrival = φ_send + delay × 2π × γ_freq / 1000
    gamma_phase = oscillator.gamma_phase
    for i, (spk, ph, dl) in enumerate(zip(incoming_spikes, incoming_phases, incoming_delays)):
        arrival_phase = ph + dl * 2 * pi * gamma_freq / 1000  # delay-corrected phase
        phase_diff = abs(arrival_phase - gamma_phase) % (2*pi)
        coincident = spk & (phase_diff < phase_window * 2*pi * gamma_freq / 1000)
        # Tylko koincydentalne spike'i (po uwzględnieniu opóźnienia) trafiają do target
        effective_input += coincident * connection_weight
```

**Bez delay correction:** Long-range Small-World "autostrady" z 5-10ms opóźnieniem trafiałyby w inną podfazę gamma (cykl @ 40Hz = 25ms → 5ms = 72° phase shift) — wiązanie wielomodalne by się rozpadło.

**Biologicznie:** Phase precession (O'Keefe & Recce 1993) — neurony hipokampalne uczą się KIEDY nadawać, żeby dotrzeć we właściwej fazie. Arrival-time coincidence jest fundamentalnym mechanizmem wiązania w mózgu.

Wiązanie odbywa się fizycznie: neurony reprezentujące "czerwony" i "okrągły" które wypalają w tej samej podfazie gamma są "zbindowane" — downstream widzi je jako jeden obiekt. Neurony "czerwony" wypalające w innej podfazie niż "kwadratowy" → nie bindowane.

Silenie przez interneurony inhibicyjne (InhibitoryPool): szybkie hamowanie lateral wymusza synchronizację w ramach grup winner-take-all.

### Krok 3.2: Biologiczna Kompetycja zamiast Softmax

**Plik:** `core/attention.py`

USUNĄĆ z `SpatialAttentionController.compute()`:

- `td_exp = np.exp(shifted / T)` — softmax
- `probs = exp_vals / sum(exp_vals)` — normalizacja softmax

ZASTĄPIĆ obwodem wzajemnej inhibicji:

```
# Każda kolumna ma dedykowany interneuron
# Excitatory drive = saliency (bottom_up + top_down)
# Inhibitory feedback = mutual inhibition from other columns

class AttentionCircuit:
    def __init__(n_columns, config):
        self.v_exc = zeros(n_columns)   # excitatory popul. per column
        self.v_inh = zeros(n_columns)   # inhibitory interneuron per column
        self.w_mutual = ones(n_columns, n_columns) - eye(n_columns)  # all-to-all inhib
        self.tau_exc = 10.0  # ms
        self.tau_inh = 5.0   # ms (faster — stabilizing)

    def step(saliency):  # saliency[i] = bottom_up_weight*PE_i + (1-bottom_up_weight)*top_down_i
        # Excitatory: driven by saliency, inhibited by others
        self.v_exc += dt/tau_exc * (-v_exc + saliency - v_inh)

        # Inhibitory: driven by total excitatory activity
        total_exc = v_exc @ w_mutual  # sum of other columns' excitation
        self.v_inh += dt/tau_inh * (-v_inh + total_exc)

        # Gain = ReLU(v_exc) — winning column has positive v_exc
        gains = maximum(v_exc, 0)
        # Normalize by total gain (divisive normalization — Reynolds & Heeger 2009)
        gains /= (sum(gains) + 1e-8)
        return gains
```

NE moduluje siłę inhibicji:

- Low NE → weak inhibition → broad attention (multiple winners)
- Optimal NE → strong inhibition → sharp focus (single winner)
- High NE → overshoot → oscillates (too much mutual inhibition)

Emergentna krzywa inverse-U (Yerkes-Dodson) BEZ jawnego kwadratowego wzoru `T = T_base × (1 + (ne - ne_opt)²)`.

### Krok 3.3: Usunięcie np.random.choice z Selekcji Akcji

**Plik:** `core/basal_ganglia.py`

W `D1D2Actor.forward()` — USUNĄĆ:

```python
# USUNĄĆ:
exp_val = np.exp(shifted)
probs = exp_val / (np.sum(exp_val) + 1e-10)
action = int(np.random.choice(self.motor_dim, p=probs))
```

ZASTĄPIĆ:

```python
# Akcja = neuron z najwyższą rate (D1 - D2 net evidence)
# Eksploracja emerguje z: membrane_noise + NE-modulated noise + VRP stochasticity
action = int(np.argmax(net_evidence))
```

**NOTA: `argmax` to TYMCZASOWY SHORTCUT łamiący P1.** Docelowo D1 populacje per action powinny konkurować przez wzajemną inhibicję — identyczny mechanizm jak `AttentionCircuit` w Kroku 3.2. Wzorzec: `v_exc` per action accumulates, `v_inh` mutual inhibition, winner = max(v_exc). `argmax` jest akceptowalnym przybliżeniem dopóki competition circuit nie jest zaimplementowany — readout jest taki sam (najwyższy v_exc ≈ argmax), różni się jedynie dynamiką konwergencji.

Ale to `argmax` — to nadal algorytm? Nie. Eksploracja emerguje fizycznie:

- `membrane_noise_std = 2.0 mV` (już istnieje w BG config)
- VRP stochastyczność (Krok 2.5): uwolnienie pęcherzykowe jest losowe → spike patterns fluktuują
- NE moduluje amplitudę szumu (już istnieje: `ne_trace_compression`)
- Adaptacja AdEx (Krok 1.1): prąd w powoduje, że frequently-fired actions się adaptują → inne dostają szansę

Przy niskim NE (exploitation): szum mały, net evidence stabilne → argmax ≈ best action.
Przy wysokim NE (exploration): szum duży, net evidence fluktuuje → różne akcje wygrywają.

W `ActiveInferenceModule.select_action()` — USUNĄĆ cały for-loop i softmax (gruntowna zmiana w Fazie 4).

### Krok 3.4: Teoria Tysiąca Mózgów (Kolumny z Reference Frames)

**Plik:** `core/columnar.py`, `core/config.py`

Dodać `GridCellConfig(BaseConfig)`:

- `n_modules: int = 3` (3 moduły grid cells per kolumna — minimalny zestaw)
- `cells_per_module: int = 8` (8 komórek per moduł)
- `module_scales: tuple = (1.0, 1.7, 2.9)` (skale gridów — stosunek ~√3, Stensola 2012)

Dodać klasę `GridCellModule`:

- Stan: `phase: NDArray[float32]` shape (n_modules, cells_per_module) — faza periodyczna
- Aktualizacja displacement: `phase += velocity_input * dt / scale` po module
- Kodowanie pozycji: `encoding = cos(phase)` — population code z periodycznym receptive field
- Uczenie: Hebbian between grid cells and sensory layer (grid phase → feature prediction)

Każda kolumna korowa (z `build_columnar_network()`) teraz składa się z:

1. **Sensory (L4):** PredictiveCodingLayer (bez zmian)
2. **Grid cells (DG/entorhinal analogue):** GridCellModule — 1D displacement tracking
3. **Object model (L2/3):** lokalne wagi rekurencyjne + Hebbian association grid⇄sensory
4. **Horizontal links:** per-column prediction → broadcast do innych kolumn

**Konsensus horyzontalny:**

- Każda kolumna generuje prediction: "myślę że to obiekt X" (swój L2/3 attractor)
- Kolumny wymieniają predictions (horizontal connections — long-range w Small-World)
- Zgodne predictions wzmacniają się; niezgodne wygaszają (mutual inhibition)
- Wynik: kolumny konwergują na wspólną interpretację (consensus = percepcja)

Początkowa implementacja: 1D displacement (dla CartPole/MountainCar — velocity integration).

### Krok 3.5: Homeostaza Wrodzona (Anti-Dark-Room)

**Pliki:** nowy `core/homeostatic_drive.py`, `core/free_energy.py`, `core/config.py`

Dodać `HomeostaticPriorConfig(BaseConfig)`:

- `energy_setpoint: float = 0.8` (docelowy poziom ATP)
- `arousal_setpoint: float = 0.3` (docelowy poziom aktywności sieci)
- `arousal_tau: float = 5000.0` (ms — EMA aktywności, ~5s)
- `prior_pe_weight: float = 2.0` (siła prediction error od priors — silny drive!)

Klasa `HomeostaticDrive`:

- Stan: `arousal_ema: float` — bieżący średni firing rate sieci
- Stan: `energy_level: float` — mean ATP z AstrocyteField

Aktualizacja per step:

```
arousal_ema = arousal_ema * decay(arousal_tau) + mean_network_rate * (1 - decay(arousal_tau))
energy_level = mean(astrocyte.atp)

# Prediction errors od priorytetów (interoceptive PE)
pe_energy = (energy_setpoint - energy_level) * prior_pe_weight
pe_arousal = (arousal_setpoint - arousal_ema) * prior_pe_weight
```

Integracja z `free_energy.py`:

```
def variational_free_energy_with_priors(
    exteroceptive_pe,    # PE od sensorów (istniejące)
    precision,            # astrocyte precision (istniejące)
    homeostatic_pe,       # PE od priors (nowe)
):
    F_extero = 0.5 * precision * sum(exteroceptive_pe²)
    F_intero = 0.5 * sum(homeostatic_pe²)  # precision=1 (rigid priors)
    return F_extero + F_intero
```

Efekt Anti-Dark-Room: jeśli agent nic nie robi → ATP się regeneruje (OK) ALE arousal spada poniżej setpoint → duży PE arousal → sygnał wymuszający aktywność. Agent nie może "zasnąć na zawsze" — wrodzone priory wyciągają go z bezruchu.

### Weryfikacja Fazy 3

1. **Phase binding with delays:** Dwa sygnały z RÓŻNYCH dystansów (delay 1ms i 5ms) zsynchronizowane po arrival-time correction: downstream odczytuje je jako 1 obiekt. Bez correction: binding fails.
2. **Attention bez softmax:** Kolumna z najwyższym PE wyłania się jako "winner" z obwodu inhibicji w <50 kroków bez softmax.
3. **Brak np.random.choice:** `grep -rn "random.choice" core/` → zero wyników.
4. **Dark room:** Środowisko dające 0 reward za nic → agent MIMO TO inicjuje akcje (homeostatic PE > 0).
5. **Grid cells:** 1D velocity integration: phase wraps correctly, object identity stable through displacement.

---

## FAZA 4: Makro-Planowanie i Sen jako Symulator

**Cel:** Wyobraźnia przez bramkowanie wzgórzowe, planowanie przez dryf atraktorowy, konsolidacja przez SWR, sen wyzwalany energetycznie.

**Zależności:** Wymaga Fazy 3 (synchronia fazowa musi działać zanim wprowadzamy bramkowanie sensoryczne).

### Krok 4.1: Bramkowanie Wzgórzowe (Thalamic Gating)

**Pliki:** `core/network.py`, `core/world_model.py`

USUNĄĆ z `core/world_model.py`:

- Klasę `EncoderSnapshot`
- Metodę `snapshot_encoder()` / `restore_encoder()`
- Mechanizm save/restore w `mental_rehearsal()`

DODAĆ do `NetworkGraph`:

- Flaga `thalamic_gate: bool = False`
- W `step()`: gdy `thalamic_gate == True`:
  - `sensory_inputs = {name: zeros_like(v) for name, v in sensory_inputs.items()}`
  - Reszta dynamiki bez zmian — identyczne wagi, identyczne interneurony, identyczny oscylator
  - Sieć "wyobraża sobie" — wewnętrzne atraktory dryfują bez zakotwiczenia sensorycznego

**KRYTYCZNE: Plastyczność podczas wyobraźni — bramkowanie różnicowe przez ACh:**

Hasselmo (2006): podczas aktywnej percepcji ACh jest WYSOKA (encoding mode). Podczas planowania/wyobraźni ACh SPADA (retrieval mode → encoding OFF).

Implementacja: gdy `thalamic_gate == True`:

```
# Wymuszamy spadek ACh na poziomie neuromodulatora
neuromodulator.set_imagination_mode(True)
# → acetylcholine = 0.1 (retrieval floor, nie zero — pozwala na weak encoding)
```

Efekty na warstwy (już istniejące mechanizmy!):

- **PredictiveCodingLayer:** `ach_level = 0.1` → combined = 0.1 × error_gradient + 0.9 × top_down → dominacja wewnętrzna
- **PyramidalLayer:** `_ach_apical_scale = 0.1` → apical-dominant (top-down context drives)
- **Neuron STDP:** ACh kompresuje membrane τ (Krok 1.1 — `ach_membrane_compression`) — przy niskim ACh membrane jest wolna → mniej spike'ów → mniej STDP

**KRYTYCZNIE NOWY MECHANIZM — ACh bezpośrednio bramkuje learning rate:**

W `AdExLayer` (Krok 1.1) rozszerzyć `set_plasticity_timescales()`:

```
# Dodać: ACh bezpośrednio skaluje learning rate (Hasselmo 2006)
# Parametry w NeuromodulatorConfig (NIE hardkodowane):
#   ach_encoding_steepness: float = 10.0  — stromość sigmoidy gate'a
#   ach_encoding_threshold: float = 0.3   — punkt przegięcia
encoding_gate = sigmoid(ach_encoding_steepness * (ach - ach_encoding_threshold))
# encoding_gate ≈ 0 gdy ACh=0.1, ≈ 1 gdy ACh=0.5+
self._learning_rate_scale = encoding_gate
```

W update_weights: `dw *= self._learning_rate_scale`

**BG plasticity NIE jest bramkowana przez ACh** — D1/D2 learning zależy od DA, nie ACh. Agent MOŻE uczyć wartości akcji z wyobrażeń (to cel planowania), ale NIE może uczyć sensorycznych skojarzeń z halucynacji.

### Krok 4.2: Planowanie jako Dryf Atraktorowy (nie for-loop)

**Pliki:** `core/basal_ganglia.py`, `core/world_model.py`

USUNĄĆ:

- `ActiveInferenceModule.select_action()` — cała metoda z for-loop po candidate_actions
- `SNNWorldModel.mental_rehearsal()` — cała metoda z iteracją po akcjach i krokach
- `ActiveInferenceModule.compute_epistemic_values()` — bazuje na mental_rehearsal

ZASTĄPIĆ jednym mechanizmem:

```
def plan(network, oscillator, n_theta_cycles=3):
    """Fala planująca: BG noise → world model drift → value read-out.

    Nie iteruje po akcjach. System SAM wpada w basen atraktorowy
    odpowiadający najlepszej akcji (gradient Expected Free Energy).

    n_imagination_steps wynika z oscylatora:
    1 theta cycle = theta_period / gamma_period = (1000/6) / (1000/40) ≈ 7 gamma steps
    n_imagination_steps = n_theta_cycles × int(theta_period / gamma_period)
    """
    theta_period = 1000.0 / oscillator.theta_freq  # ~167ms
    gamma_period = 1000.0 / oscillator.gamma_freq  # ~25ms
    n_imagination_steps = n_theta_cycles * int(theta_period / gamma_period)  # 3 × 6 = 18
    network.thalamic_gate = True
    neuromod.set_imagination_mode(True)

    # BG generuje diffuse noise (thalamocortical stochasticity)
    noise_amplitude = neuromod.competition_sharpness  # NE-modulated

    for _ in range(n_imagination_steps):
        # Noise injection into BG → propagates to world model
        bg_noise = randn(bg_input_size) * noise_amplitude * exploration_noise

        # Single network step — world model drifts through attractor landscape
        network.step(sensory_inputs={}, bg_noise=bg_noise)

        # Oscillator ticks — gamma resets trigger phase binding
        # Network naturally settles toward low-EFE attractor

    # Read out: which action basin did we land in?
    action = argmax(actor.net_evidence)  # D1 - D2

    network.thalamic_gate = False
    neuromod.set_imagination_mode(False)
    return action
```

System nie "oblicza" optymalnej akcji — **spływa** do niej po gradiencie Expected Free Energy. Atraktor z najniższym G(a) ma najgłębszy basen → system najczęściej tam trafia.

Przy wyższym NE: szum silniejszy → system eksploruje więcej atraktorów.
Przy niższym NE: szum słaby → system tkwi w lokalnym minimum.

### Krok 4.3: Hierarchiczne Delegowanie

**Pliki:** `core/sequence_memory.py`, `core/basal_ganglia.py`

`HierarchicalSequenceMemory` level 2 (theta) już istnieje ale nie outputuje makro-celów.

Dodać:

1. **Macro-goal output z Level 2:**

   ```
   def predict_next_goal(self) -> NDArray[float32]:
       """Na podstawie theta-pooled transitions, przewidź następny pattern."""
       return self.levels[2].predict(current_pattern)
   ```

2. **BG veto/approve:**

   ```
   # Macro-goal → BG input
   # D1 (Go): jeśli expected_value(goal) > threshold → approve
   # D2 (NoGo): jeśli expected_cost(goal) > threshold → veto
   # Net evidence > 0: approve → przekaż do niższych warstw
   # Net evidence < 0: veto → odrzuć, sequence memory proponuje alternatywę
   ```

3. **Motor delegation:** Zaakceptowany makro-cel → top-down prediction do kolumn → kolumny autonomicznie generują mikro-akcje (gamma-level reflexes).

### Krok 4.4: Sharp-Wave Ripples (SWRs) w Konsolidacji Snu

**Plik:** `core/replay_buffer.py`

USUNĄĆ:

- Iteracyjny Python for-loop w `_sws_phase()` po `reversed(experiences)`
- Ręczne przywracanie eligibility traces z `synaptic_fingerprint`

ZASTĄPIĆ konsolidacją SWR:

```
def _sws_phase(experiences, network, oscillator, ewc_manager):
    """SWS: Sharp-Wave Ripples z CANN pamięci epizodycznej.

    1. Pobudź CANN (SDM) szumem → sieć relaksuje do wyuczonego atraktora
    2. Atraktor = zakodowane wspomnienie → burst propaguje wstecz
    3. TD error z re-experienced memory → BG value update
    4. Eligibility during SWR → Fisher information → EWC stiffness
    """
    oscillator.enter_sws()

    for ripple in range(n_ripples):
        up_onset, down_onset = oscillator.tick_sws()

        if not oscillator.in_up_state:
            continue  # Down state: cisza

        # 1. CANN recall: random partial cue → relax to stored pattern
        cue = random_partial_cue(experiences)
        recalled = episodic_memory.recall(cue)

        # 2. Propagacja: recalled pattern → forward through network
        #    (thalamic_gate = False — SWR drives are internal but treated as "real")
        #    ACh is LOW during SWS → cortical STDP gated
        #    BUT: EWC stiffness updates ARE active (consolidation-specific)
        network.step(sensory_inputs=recalled_as_input)

        # 3. TD error from recalled experience
        recalled_value = critic.last_value
        td_swr = recalled.reward + gamma * recalled_value - prev_value
        critic.update(td_swr)
        actor.update(td_swr)

        # 4. EWC: accumulate Fisher information from SWR gradients
        #    KRYTYCZNE: Fisher information = WARIANCJA gradientu, nie suma.
        #    Bez kwadratu: dodatnie i ujemne gradienty z różnych ripples
        #    zerują się → sztucznie niska stiffness → brak ochrony wiedzy.
        for layer in network.layers_with_weights:
            grad_proxy = layer.e * td_swr  # eligibility × TD error
            ewc_manager.accumulate_fisher(grad_proxy ** 2)  # KWADRAT!

    # Zatwierdź nową stiffness
    ewc_manager.commit_fisher()
    oscillator.exit_sws()
```

Kompresja czasowa: 1 SWR ≈ 200ms real, odtwarza sekundy/minuty doświadczenia. ~50 SWR per "noc" = pokrycie kluczowych doświadczeń.

### Krok 4.5: Sen Wyzwalany Energią (nie Epizodem)

**Plik:** `arena/snn_agent.py`, `core/config.py`

Dodać do `AgentConfig`:

- `sleep_atp_threshold: float = 0.3` (śpij gdy mean ATP < 30%)
- `max_wake_duration_ms: float = 300_000.0` (5 minut max bez snu)
- `sleep_duration_base_ms: float = 10_000.0` (10s bazowy sen)

USUNĄĆ z `observe()`:

- Blok `if done and self._use_wm and len(self.replay_buffer) > 0:` (sleep at episode end)

ZASTĄPIĆ ciągłą oceną:

```
def _check_sleep_need(self) -> bool:
    """Czy system potrzebuje snu? Decyzja termodynamiczna."""
    mean_atp = float(np.mean(self.astrocyte.atp))
    time_awake = self._step_count - self._last_sleep_step

    return (mean_atp < sleep_atp_threshold) or (time_awake * dt > max_wake_duration_ms)

def _sleep(self):
    """Faza snu — odpalana gdy _check_sleep_need() == True."""
    sleep_duration = sleep_duration_base_ms * (1 - mean_atp / atp_max)  # dłuższy sen przy większym deficycie
    n_ripples = int(sleep_duration / 200)  # ~200ms per SWR

    self.replay_buffer.swr_consolidation(
        network=self.network,
        oscillator=self.network.oscillator,
        ewc_manager=self.consolidation,
        episodic_memory=self.episodic_memory,
        n_ripples=n_ripples,
    )

    # Regeneracja ATP podczas snu (regen_rate wyższy niż w wake)
    self.astrocyte.sleep_regeneration(duration_ms=sleep_duration)
    self._last_sleep_step = self._step_count
```

Cykl: wake → ATP spada → próg rośnie → sieć cichnie → sleep trigger → SWR konsolidacja + ATP regen → wake refreshed.

### Weryfikacja Fazy 4

1. **Thalamic gate:** Sensory = zeros, internal dynamics active, ACh < 0.15. Wagi korowe NIE zmieniają się (STDP gate ≈ 0). Wagi BG ZMIENIAJĄ się (DA-driven).
2. **Wave planning:** Bez for-loop: 20 imagination steps → action = argmax(net_evidence). Akcja ≥ 80% zgodna z for-loop baseline.
3. **SWR:** 50 ripples → EWC stiffness wzrasta na krytycznych wagach. BG value function updates.
4. **Energy sleep:** Ciągła symulacja 10 000 kroków → agent sam zasypia ~3 razy, budzi się z wyższym ATP.
5. **No hallucination learning:** Przed/po 1000 imagination steps: korowe wagi zmiana < 1%. BG wagi zmiana > 5%.

---

## FAZA 5: Integracja i Walidacja Końcowa

### Test 5.1: Ciągłe Uczenie (Continual Learning)

- Agent na sekwencji: CartPole → MountainCar → CartPole (powrót)
- Po powrocie do CartPole: score > 70% oryginalnego (EWC chroni)
- Brak explicit task switches z perspektywy agenta

### Test 5.2: Skalowalność

- Sieć 10 000 neuronów, 100 000 synaps (sparse)
- Pamięć < 100 MB (vs. ~800 MB przy dense 10K×10K float32)
- Step time < 10ms na CPU

### Test 5.3: Kompozycja (Zero-Shot Transfer)

- Nauczyć cechy A i cechę B oddzielnie
- Pokazać A+B jednocześnie (nowy obiekt)
- Agent binduje A+B przez gamma synchronię → transferuje wiedzę o A i B na nowy kompozyt

### Test 5.4: Anti-Dark-Room

- Środowisko: nic nie robienie daje 0 reward, 0 surprise
- Agent MUSI inicjować akcje (homeostatic PE arousal > 0)
- Verify: mean action rate > 0.3 per timestep

### Test 5.5: Naturalny cykl snu

- 100 000 kroków ciągłej pracy (bez episodów)
- Agent samodzielnie przechodzi cykle wake→sleep→wake
- Performance po śnie > performance tuż przed snem (konsolidacja działa)

---

## Podsumowanie Zależności

```
Faza 0 (higiena) → obowiązkowa
    ↓
Faza 1 (biofizyka) → AdEx, Exp Euler, ATP, DA integrator, bez epizodów
    ↓
Faza 2 (topologia) → Sparse, Structural Plasticity, SDM, EWC, VRP
    ↓
Faza 3 (kognicja) → Phase Binding, Attention Circuit, 1000 Brains, Priors
    ↓
Faza 4 (planowanie) → Thalamic Gate, Wave Planning, SWR, Energy Sleep
    ↓
Faza 5 (walidacja) → End-to-end testy
```

Każda faza kończy się weryfikacją gatekeepingową — nie przechodzić dalej bez zielonych testów.

## Wykluczone z Zakresu

- Móżdżek (cerebellar motor refinement) — slot architektoniczny zostawiony
- Pełne 3D grid cells — zaczynamy od 1D displacement
- Multi-agent communication
- Kompilacja sprzętowa (neuromorphic hardware)
- Uczenie ze wzmocnieniem w środowiskach wizualnych (image input)
