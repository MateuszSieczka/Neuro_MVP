# PLAN NAPRAWY: Diagnostyka Porażki i Ścieżka do Uczącego się SNN

## Status: Po Fazie 0 + 1, sieć nie uczy się (mean=21.8 ≈ random) nawet po 2000 epizodów.

---

## 0. PEŁNA DIAGNOSTYKA — WYNIKI ILOŚCIOWE

### 0.1 Architektura CartPole (flat mode, bez WM/columnar)

```
State (4D) → GaussianPopulationEncoder (4×15=60D, sigma=0.5×spacing)
  → PoissonEncoder (60D binary per substep)
  → [25 substeps] → D1D2Actor + SNNDeepCritic (AdEx, membrane potential readout)
  → get_action() → argmax(D1_accum - D2_accum + noise)
  → observe(): [15 substeps critic-only] → V(s') → TD error → 3-factor STDP update
```

- `n_substeps_act = 25` (= τ_m_msn_up/dt = 25ms/1ms)
- `n_substeps_observe = 15` (= τ_m_critic/dt = 15ms/1ms)
- `critic_input_gain = 125.76`, `actor_input_gain = 377.28` (from AdEx rheobase)
- `neurons_per_action = 32`, `hidden_size = 128`
- `gamma = 0.95`, `critic_lr = 0.001`, `actor_lr = 0.01`

### 0.2 Przepływ Danych — Co Się Łączy z Czym

```
act(state):
  1. pop_rates = GaussianPopEncoder.encode(state)           # 4D → 60D [0,1]
  2. for 25 substeps:
       encoded = PoissonEncoder.encode(pop_rates)           # 60D binary
       sensory = {"critic": encoded, "actor": encoded}
       network.step(sensory)                                # → critic.forward() + actor.forward()
  3. _set_actor_policy_gradient(pop_rates)   ← REINFORCE OVERRIDE (BUG #1)
  4. action = actor.get_action()

observe(s, a, r, s', done):
  1. Save critic + actor eligibility
  2. for 15 substeps:
       critic-only step on next_state
  3. V(s') = critic.last_value
  4. Restore eligibility
  5. TD = r + γ*V(s') - V(s)  [clipped to ±10]
  6. critic.update(TD), actor.update(TD)
  7. neuromod.update(td_error)
```

### 0.3 Wyniki Diagnostyczne (skrypty: `_diag_pipeline.py`, `_diag_learning.py`, `_diag_normalization.py`, `_diag_gated_stdp.py`)

---

## 1. ROOT CAUSES — Dlaczego Sieć Nie Uczy Się

### ROOT CAUSE #1: REINFORCE Policy Gradient Override [KRYTYCZNY, BUG]

**Dowód:**
```
STDP eligibility Frobenius norm:  26.07
PG override Frobenius norm:       0.12
Ratio: 216× magnitude reduction
```

**Mechanizm:** W `snn_agent.py`, po pętli substepów w `act()`, metoda `_set_actor_policy_gradient(pop_rates)` ZASTĘPUJE biologicznie-zbudowane eligibility traces z softmax policy gradient:

```python
# Z _set_actor_policy_gradient():
scale = (indicator - probs[a]) / n_per_action   # ÷32 = 0.031 max
col = inp * scale                                 # outer product z pop_rates
actor.e_d1[:, start:end] = col[:, np.newaxis]    # NADPISUJE eligibility
```

**Dlaczego to jest problem:**
1. `n_per_action = 32` → dzieli skalę przez 32
2. `pop_rates` ma mean=0.08 (większość ~0) → outer product jest ultra-sparse
3. Wynik: `|e_d1|_mean = 0.0004` vs STDP `|e_d1|_mean = 0.10`
4. Efektywny update per step: `lr × TD × |e| = 0.01 × 1.0 × 0.0004 = 0.000004`
5. Per 200 epizodów (4000 kroków): `Δw ≈ 0.025` — niezauważalne

**Konsekwencje:**
- Waga floor (0.01) generuje `Δw ≈ 0.000337/step` — **84× więcej** niż learning signal
- Homeostaza generuje 1.49× over 200 eps — dominated by scaling, not learning
- Actor weights stay nearly identical for both actions (cosine=0.985)
- Actions are noise-dominated (SNR=0.03-0.12)

**To jest hack ML (softmax REINFORCE gradient), nie biologiczny mechanizm. Narusza P1 (fizyka zamiast algorytmu).**

**Literaturowy kontrargument:** Eligibility traces w SNNs powinny emergować z dynamiki spike-timing (Bi & Poo 2001), nie z analitycznego ∇log π. REINFORCE jest algorytmem stochastic gradient, nie procesem biologicznym.

---

### ROOT CAUSE #2: Brak `gate_eligibility()` — Oba Actions Aktualizowane Identycznie [KRYTYCZNY]

**Dowód:**
```
e_d1 Action 0: |mean| = 0.164 (STDP without gating)
e_d1 Action 1: |mean| = 0.188 (STDP without gating)
→ OBA actions mają eligibility, update zmienia OBA jednakowo
```

**Mechanizm:** Metoda `gate_eligibility(action)` **istnieje** w `D1D2Actor`, ale **nigdy nie jest wywoływana**. Bez gatingu:
- TD > 0 (19/20 kroków): D1 rośnie dla OBU akcji, D2 maleje dla OBU
- TD < 0 (1/20 kroków): odwrotnie, ale dla OBU
- Net effect: obie akcje ewoluują identycznie → brak różnicowania

**Biologiczne uzasadnienie gatingu:** Redgrave, Gurney & Reynolds (2010): DA reinforcement jest skierowane na synapsy WYBRANEJ ścieżki action channel w striatum. Non-selected MSN pools nie otrzymują phasic DA burst (poprzez local feedback inhibition i dendritic gating).

---

### ROOT CAUSE #3: Addytywny Temperature Noise Niszczy Sygnał [WAŻNY, HACK]

**Dowód:**
```
Net evidence mean:     [-0.121, -0.030]  (signal)
Net evidence std:       [0.222, 0.245]    (intrinsic SNN noise)
Added temperature noise: N(0, ~1.0)
Effective SNR:          [0.118, 0.029]    ← almost zero
```

**Mechanizm:** Po normalizacji net evidence, dodawany jest **osobny** noise z rozkładu `N(0, temperature)`:
```python
temperature = 1.0 + 4.0 * (self._ne_level - 0.5) ** 2  # ≈ 1.0-1.16
noise = np.random.normal(0.0, temperature, self.motor_dim)
noisy_evidence = net_evidence + noise
action = argmax(noisy_evidence)
```

**Dlaczego to jest hack:**
1. SNN **już generuje trial-to-trial variability** z Poisson input, membrane noise (1 mV), i stochastic spiking. To jest biologicznie poprawne źródło eksploracji (Faisal, Selen & Wolpert 2008: "Noise in the nervous system").
2. Dodatkowy addytywny noise jest **podwójnym liczeniem szumu** — biologically there's no separate "exploration noise generator" in the brain
3. Usher & Damasio (2000) model NE jako **multiplicative gain modulation**, nie additive noise. NE zmienia gain neuronów (ostrość odpowiedzi), nie dodaje szumu do outputu.

**Poprawna implementacja NE-modulation (Usher & Damasio 2000, Aston-Jones & Cohen 2005):**
NE moduluje gain WEJŚCIA do MSN (multiplicative), nie noise na WYJŚCIU (additive):
```python
# CURRENT (hack): evidence + additive_noise
# CORRECT: gain_modulated_current → naturally variable evidence
ne_gain = 1.0 / (1.0 + 4.0 * (ne - 0.5)**2)  # Inverse-U gain
current_d1 *= ne_gain
current_d2 *= ne_gain
# No added noise — SNN variability IS the exploration
```

---

### ROOT CAUSE #4: Net Evidence Normalization Jest Zbyt Agresywna [WAŻNY]

**Dowód:**
```
Raw D1-D2 difference:  ~20-80 (meaningful)
After / (n_per_action × √n_substeps) = / 160: ~0.1-0.5
Temperature noise scale: ~1.0
→ signal / noise = 0.1
```

**Mechanizm:**
```python
_norm = self.n_per_action * np.sqrt(float(_n))  # 32 × 5 = 160
net_evidence = (motor_d1 - motor_d2) / _norm
```

**Problem:** Normalizacja dzieli przez `32×√25 = 160`, sprowadzając evidence do skali ~0.1. Komentarz mówi: "signal grows linearly with substeps but noise grows as √substeps, so SNR ∝ √N". To jest statystycznie poprawne, ALE konkluzja "divide by n_per_action × √n_substeps gives unit-SNR evidence" jest BŁĘDNA — to nie daje unit-SNR, to NISZCZY evidence scale.

**Poprawna normalizacja:** Evidence accumulation w drift-diffusion model (Gold & Shadlen 2007) NIE normalizuje przez czas — sygnał akumuluje się, a decyzja zapada gdy sygnał przekroczy threshold. Normalizacja powinna zachować rosnący SNR:

```python
# Membrane potential averages: divide only by n_substeps to get mean voltage
v_d1_mean = v_accum_d1 / n_substeps  # Mean per-neuron voltage [0,1]
v_d2_mean = v_accum_d2 / n_substeps
# Population mean difference (not sum → mean avoids n_per_action scaling)
motor_d1 = v_d1_mean[:total_motor].reshape(motor_dim, n_per_action).mean(axis=1)
motor_d2 = v_d2_mean[:total_motor].reshape(motor_dim, n_per_action).mean(axis=1)
net_evidence = motor_d1 - motor_d2  # Scale: ~[-0.3, 0.3] naturally
```

---

### ROOT CAUSE #5: InhibitoryPool Jest Martwy [WAŻNY, STRUKTURALNY]

**Dowód:**
```
critic pool:   v_inh mean=-69.01 (rest=-70), 0 spikes, 0 GABA
actor_d1 pool: v_inh mean=-69.54, 0 spikes, 0 GABA
actor_d2 pool: v_inh mean=-69.87, 0 spikes, 0 GABA
```

**Root cause:** Gain kalibrowany na `target_sparsity × N` active inputs, ale actual firing jest dużo niższe:

| Pool | target_sparsity | Expected active | Actual active | Input ratio |
|------|----------------|-----------------|---------------|-------------|
| critic | 0.15 | 19.2 | ~3.6 (2.8%) | 0.19× |
| actor_d1 | 0.05 | 3.2 | ~0.04 (0.06%) | 0.01× |
| actor_d2 | 0.05 | 3.2 | ~0.08 (0.12%) | 0.03× |

Interneurony dostają 1-19% oczekiwanego wejścia → daleko poniżej rheobase.

**Dodatkowy bug: g_L mismatch:** InhibitoryPool tworzy `NeuronConfig(tau_m=8)` ale NIE nadpisuje `g_L`. Default `g_L=30 nS` z `C_m=281 pF` daje efektywne `τ_m = C_m/g_L = 9.37 ms`, nie 8 ms. AdEx step używa `g_L` bezpośrednio, więc `tau_m` wpływa tylko na legacy `mem_decay`.

**Konsekwencje braku inhibicji:**
- Brak lateral competition → brak winner-take-all → brak selekcji
- Brak E/I balance → wagi rosną bez kontroli
- Brak sparsification → nadmiarowa aktywność w critic nie jest ograniczona

---

### ROOT CAUSE #6: Weight Floor (0.01) Dominuje nad Learning Signal [UMIARKOWANY]

**Dowód:**
```
Weight floor Δw per step: 0.000337 (283/3840 entries reset to 0.01)
PG learning Δw per step:  0.000004
Ratio: 84× — floor > learning
```

Floor `np.maximum(self.w_d1, 0.01)` w `D1D2Actor.update()` jest stosowany po KAŻDYM update. Z dużą normą eligibility (STDP) to nie byłby problem (learning >> floor), ale z override'm floor dominuje.

**Biologiczne uzasadnienie floor:** Kerchner & Nicoll (2008): NMDA-only silent synapses. Ale 0.01 jest za duże — silent synapses mają effectively zero AMPA conductance, nie 10% nominal weight.

---

### ROOT CAUSE #7: b_v Bias Rośnie Monotonicznie [UMIARKOWANY]

**Dowód:**
```
b_v: 0 → 8.4 (ep 25) → 12.9 (ep 200)
V(balanced): 14.1 = 12.9 (b_v) + 1.2 (state-dependent)
```

**Mechanizm:** `e_bv = e_bv × decay + v_mean_pop` gdzie `v_mean_pop ≈ 0.5-0.7` (always positive). Z `readout_decay = 1e-5`, b_v prawie nie rozpada się.

**Problem:** b_v inflates V(s) → inflates terminal TD error → TD clip (-10) atenuuje. V(s) ≈ 14 ale true V ≈ 13 (computed analytically for 20-step random episodes with γ=0.95). Niewielki mismatch, ale b_v uniemożliwia szybkie dostosowanie V(s) baseline.

---

## 2. LISTA HACKÓW I UPROSZCZEŃ DO USUNIĘCIA

### 2.1 HACK: `_set_actor_policy_gradient()` — Softmax REINFORCE [ROOT CAUSE #1]
**Linia:** `arena/snn_agent.py` method `_set_actor_policy_gradient`
**Co to jest:** Analityczny ∇log π(a|s) gradient z softmax policy — standard ML REINFORCE
**Czemu problematyczny:** Nie jest biologiczny. Zastępuje spike-timing eligibility traces. Niszczy magnitude 216×.
**Usunąć:** TAK — zastąpić biologicznym gate_eligibility + STDP

### 2.2 HACK: Temperature noise na net evidence [ROOT CAUSE #3]
**Linia:** `core/basal_ganglia.py` D1D2Actor.forward() ~line 1040
**Co to jest:** `noise = normal(0, temperature, motor_dim)` dodane do evidence
**Czemu problematyczny:** Podwójne liczenie szumu; biologicznie NE to multiplicative gain, nie additive noise
**Usunąć:** TAK — zamienić na NE multiplicative input gain modulation

### 2.3 HACK: `softmax` diagnostyczny w actor.forward()
**Linia:** `core/basal_ganglia.py` ~line 1055
**Co to jest:** `shifted = noisy_evidence - max; exp; probs = exp/sum` — softmax probability
**Status:** Tylko diagnostyczny (`_last_probs`), NIE używany do selekcji akcji (argmax jest). AKCEPTOWALNE jako diagnostic.

### 2.4 UPROSZCZENIE: Evidence normalization / (n_per_action × √n_substeps)
**Linia:** `core/basal_ganglia.py` ~line 1030
**Status:** Niepotrzebne jeśli temperature noise usunięty — intrinsic variability jest na poprawnej skali

### 2.5 UPROSZCZENIE: Weight floor `np.maximum(w, 0.01)` zamiast biofizycznej silent synapse
**Linia:** `core/basal_ganglia.py` D1D2Actor.update() ~line 1140
**Status:** Zmienić floor na 0.001 lub implementację soft decay

### 2.6 BRAKUJĄCY KOD: `gate_eligibility()` nigdzie nie wywoływane
**Linia:** `core/basal_ganglia.py` — method exists but never called
**Status:** Dodać wywołanie w `act()` po pętli substepów

---

## 3. PLAN NAPRAWY (Priorytetyzowany)

### Zasady:
- **Każda zmiana ma literaturę** — papier lub derywację biofizyczną
- **Testujemy po każdej zmianie** — diagnostic scripts weryfikują efekt
- **Nie optymalizujemy pod CartPole** — naprawiamy fundamenty

---

### FIX 1: Usunięcie REINFORCE Override + Dodanie Gate Eligibility [KRYTYCZNY]

**Problem:** `_set_actor_policy_gradient()` jest non-biological hack niszniczący learning 216×.

**Zmiana:**
1. Usunąć metodę `_set_actor_policy_gradient()` z `snn_agent.py`
2. Usunąć wywołanie w `act()` (linia ~450)
3. Dodać `self.actor.gate_eligibility(action)` po pętli substepów w `act()`

**Musi zostać:**
- STDP eligibility z voltage-centered outer product (biologicznie poprawne: Clopath et al. 2010)
- `gate_eligibility()` zeruje non-selected actions (Redgrave et al. 2010)

**Efekt:** Learning magnitude rośnie 216×. Per step: `0.01 × 1.0 × 0.16 = 0.0016`. Per episode: 0.032. Po 200 eps: weights change by ~6.4 (bounded by column norm safety).

**Pliki:** `arena/snn_agent.py` (act method), can delete `_set_actor_policy_gradient()` method

**Weryfikacja:**
- `_diag_normalization.py`: |dw_d1| per step should be ~0.001 (1000× increase)
- `_diag_learning.py`: cosine(w_d1[a0], w_d1[a1]) should drop below 0.95 after 50 eps
- CartPole score trend should be visible (upward)

---

### FIX 2: Usunięcie Additive Temperature Noise → NE Multiplicative Gain [KRYTYCZNY]

**Problem:** Addytywny noise N(0, ~1) na evidence ~0.1 daje SNR=0.1. Podwójne liczenie szumu.

**Zmiana w `D1D2Actor.forward()`:**

```python
# STARY (hack: additive noise)
temperature = 1.0 + 4.0 * (self._ne_level - 0.5) ** 2
noise = np.random.normal(0.0, temperature, self.motor_dim)
noisy_evidence = net_evidence + noise
action = int(np.argmax(noisy_evidence))

# NOWY (biological: NE modulates input gain, intrinsic SNN noise provides exploration)
# Usher & Damasio (2000), Aston-Jones & Cohen (2005): inverse-U NE gain
# At optimal NE(0.5): gain=1 → sharp, focused. At extremes: gain<1 → blurry, exploratory.
# No separate noise — SNN membrane noise + Poisson input provide exploration.
action = int(np.argmax(net_evidence))
```

**NE gain modulation (przenieść wcześniej w forward()):**
```python
# Inverse-U gain applied to MSN input currents:
ne_gain = 1.0 / (1.0 + 4.0 * (self._ne_level - 0.5) ** 2)
current_d1 *= ne_gain
current_d2 *= ne_gain
```

**Uzasadnienie:** Servan-Schreiber, Printz & Cohen (1990): NE modulates neural gain (slope of input-output function). Aston-Jones & Cohen (2005): LC-NE system implements exploration-exploitation trade-off via gain modulation, not additive noise.

**Pliki:** `core/basal_ganglia.py` (D1D2Actor.forward)

**Weryfikacja:**
- Without added noise, action should be deterministic given same input → check via repeated act() calls with same state
- Exploration should come from Poisson variability (different spike patterns per trial) → different actions on repeated trials
- `_diag_normalization.py`: effective SNR should be ~1-3 (signal vs intrinsic noise)

---

### FIX 3: Naprawienie Net Evidence Normalization [WAŻNY]

**Problem:** Dzielenie przez `n_per_action × √n_substeps = 160` sprowadza evidence do ~0.1.

**Zmiana:**
```python
# STARY
_norm = self.n_per_action * np.sqrt(float(_n))
net_evidence = (motor_d1 - motor_d2) / _norm

# NOWY — population MEAN voltage difference (Gold & Shadlen 2007)
# Divide accumulators by n_substeps to get time-averaged voltage,
# then take mean across the population.  This gives evidence on
# the natural voltage scale [0, 1] per neuron, independent of
# population size or decision window length.
_n = max(self._n_forward, 1)
motor_d1 = self._v_accum_d1[:self._total_motor].reshape(
    self.motor_dim, self.n_per_action,
).mean(axis=1)  / _n   # mean voltage per action
motor_d2 = self._v_accum_d2[:self._total_motor].reshape(
    self.motor_dim, self.n_per_action,
).mean(axis=1)  / _n
net_evidence = motor_d1 - motor_d2   # Scale: [-1, 1] naturally
```

**Uzasadnienie:** Drift-diffusion model (Gold & Shadlen 2007): evidence accumulates over time, decision threshold is fixed. Mean population voltage is the biological readout of action preference (Cisek 2007).

**Pliki:** `core/basal_ganglia.py` (D1D2Actor.forward)

---

### FIX 4: Naprawienie InhibitoryPool [WAŻNY, STRUKTURALNY]

**Problem:** Gain kalibrowany na target_sparsity ale actual firing jest 5-100× niższe.

**Zmiana — dynamic gain w `InhibitoryPool.__init__`:**

```python
# STARY — uses expected rate that may mismatch reality
expected_active_exc = max(1.0, n_excitatory * cfg.target_sparsity)
self._input_gain = float(i_rheo_inh / (expected_active_exc * cfg.w_ei_mean))

# NOWY — pessimistic: assume only 1-2 concurrent spikes
# PV+ interneurons need strong single-spike sensitivity
# (Gabernet, Jadhav & Bhatt 2005: single cortical spike can trigger PV+)
# Use min 1 concurrent spike for gain derivation
expected_active_exc = max(1.0, min(3.0, n_excitatory * cfg.target_sparsity))
self._input_gain = float(i_rheo_inh / (expected_active_exc * cfg.w_ei_mean))
```

**Dodatkowy fix: g_L consistency:**
```python
# In InhibitoryPool.__init__, derive g_L from tau_m consistently:
_g_L_inh = 281.0 / cfg.tau_m_inh  # = 35.125 nS for tau=8ms
self._ncfg = NeuronConfig(
    ctx=cfg.ctx,
    v_rest=cfg.v_rest,
    v_thresh=cfg.v_thresh,
    v_reset=cfg.v_reset,
    tau_m=cfg.tau_m_inh,
    g_L=_g_L_inh,  # <-- ADD THIS
    a=0.0, b=0.0,
)
```

**Pliki:** `core/interneuron.py`

**Weryfikacja:**
- `_diag_pipeline.py`: InhibitoryPool spikes > 0
- GABA_a, GABA_b > 0

---

### FIX 5: Zmniejszenie Weight Floor [UMIARKOWANY]

**Problem:** Floor=0.01 generuje Δw=0.000337/step, 84× więcej niż PG learning.

**Zmiana:**
```python
# STARY
_W_FLOOR = 0.01
np.maximum(self.w_d1, _W_FLOOR, out=self.w_d1)

# NOWY — silent synapse floor much lower (Kerchner & Nicoll 2008)
_W_FLOOR = 1e-4
np.maximum(self.w_d1, _W_FLOOR, out=self.w_d1)
```

**Uzasadnienie:** Silent synapses (Kerchner & Nicoll 2008) mają NMDA-only conductance ~1% of mature AMPA. Floor=0.0001 zachowuje structural connectivity bez significant bias.

**Pliki:** `core/basal_ganglia.py` (D1D2Actor.update)

---

### FIX 6: b_v Decay lub Usunięcie Bias [UMIARKOWANY]

**Problem:** b_v rośnie od 0 do 13 bez ograniczeń.

**Opcja A (minimalna):** Zwiększyć readout_decay dla b_v:
```python
# W SNNDeepCritic.update():
self.b_v *= (1.0 - cfg.readout_decay * 100)  # 100× faster decay for bias
```

**Opcja B (czysta):** Usunąć b_v entirely. V(s) powinno emergować z populacji, nie z learnable bias.
```python
# Usunąć b_v z __init__, forward(), update(), reset_state()
# V(s) = dot(w_v, v_mean) — bez bias
```

**Uzasadnienie opcji B:** W biologii nie ma "neuronu bias" — wartość stanu jest zakodowana w aktywności populacji. Średnia wartość V(s) emerguje z rozkładu wag w_v, nie z osobnego parametru.

**Pliki:** `core/basal_ganglia.py` (SNNDeepCritic)

**UWAGA:** Opcja B może spowolnić początkowe uczenie się (V(s) musi zbudować baseline z samych wag). Opcja A jest bezpieczniejsza.

---

### FIX 7: TD Clip Scaling [MAŁY]

**Problem:** Clip=-10 atenuuje terminal signal gdy V(s)>11.

**Zmiana:**
```python
# Adaptive clip based on current V magnitude (Tobler 2005: adaptive DA coding)
_natural_clip = max(10.0, 2.0 * abs(prev_v) + 5.0)
td_error = float(np.clip(td_error, -_natural_clip, _natural_clip))
```

**Pliki:** `arena/snn_agent.py` (observe)

---

## 4. KOLEJNOŚĆ IMPLEMENTACJI

### Faza A: Przywrócenie Uczenia (agent > random, trend rosnący)

| # | Fix | Zmiana | Ryzyko |
|---|-----|--------|--------|
| 1 | FIX 1 | Usunąć REINFORCE override + dodać gate_eligibility | Niskie — usunięcie hack'u |
| 2 | FIX 2 | Usunąć additive noise → NE multiplicative gain | Niskie — usunięcie hack'u |
| 3 | FIX 3 | Fix net evidence normalization (mean zamiast sum/norm) | Niskie — czysto numeryczne |
| 4 | FIX 5 | Weight floor 0.01 → 0.0001 | Niskie — 1 linia |
| 5 | **RUN DIAGNOSTIC** | Sprawdź czy learning signal jest widoczny | — |

### Faza B: Stabilizacja i Infrastruktura

| # | Fix | Zmiana | Ryzyko |
|---|-----|--------|--------|
| 6 | FIX 4 | Naprawić InhibitoryPool (gain + g_L) | Średnie — zmiana dynamiki |
| 7 | FIX 6 | b_v decay (opcja A) | Niskie |
| 8 | FIX 7 | Adaptive TD clip | Niskie |
| 9 | **RUN DIAGNOSTIC** | Pełny test 250 eps | — |

### Faza C: Optymalizacja (tylko jeśli < 200 po Fazie B)

| # | Zmiana | Uzasadnienie |
|---|--------|-------------|
| 10 | Tuning gamma (0.95 → 0.99) | Jeśli V(s) nie propaguje wystarczająco |
| 11 | Tuning critic_lr (0.001 → 0.003) | Jeśli V(s) uczy się za wolno |
| 12 | n_per_action tuning (32 → 16) | Jeśli evidence jest za smooth |

---

## 5. METRYKI WERYFIKACJI PO KAŻDEJ FAZIE

| Metryka | Obecny | Po A | Po B | Cel |
|---------|--------|------|------|-----|
| Mean score (200 ep) | 21.8 | ≥ 30 | ≥ 80 | ≥ 200 |
| STDP vs PG |e| ratio | 216× | 1× (STDP only) | 1× | 1× |
| Actor weight cosine(a0,a1) | 0.985 | < 0.95 | < 0.85 | < 0.7 |
| Net evidence SNR | 0.03 | > 0.5 | > 1.0 | > 2.0 |
| InhibitoryPool active | NO | NO | YES | YES |
| V(s0)-reward corr | -0.005 | > 0.1 | > 0.3 | > 0.5 |
| V(bal)-V(fall) discrimination | 7.3 | > 5 | > 10 | > 15 |

---

## 6. CO NIE JEST PROBLEMEM (nie zmieniać)

1. **AdEx model** — poprawnie implementowany, biofizycznie spójny
2. **Exponential Euler integrator** — A-stabilny, poprawny
3. **GaussianPopulationEncoder** z sigma=0.5 — poprawne dla 15 neuronów/dim
4. **PoissonEncoder** — standard bridge rate→spike
5. **Voltage-based eligibility (centered)** — biologicznie poprawne (Clopath et al. 2010)
6. **D1/D2 update rules** — poprawne Frank (2005) Go/NoGo
7. **Dale's law enforcement** — poprawne
8. **Critic V(s) readout via membrane potential** — poprawne (Priebe & Ferster 2008)
9. **Eligibility save/restore w observe()** — poprawne (Schultz 1997)
10. **Three-factor STDP framework** — poprawny Bi & Poo (2001)
11. **Neuromodulator 4-channel system** — farmakologicznie poprawny
12. **Column norm safety bound** — rozsądne safety

---

## 7. DiAGNOSTIC SCRIPTS DO URUCHOMIENIA

Skrypty diagnostyczne (już istniejące + nowe):

| Skrypt | Co sprawdza |
|--------|-------------|
| `_diag_pipeline.py` | Pełny single-step trace: gains, currents, spikes, V, eligibility, InhPool |
| `_diag_learning.py` | 200-ep learning curve: V(s) discrimination, weight evolution, cosine |
| `_diag_normalization.py` | Net evidence SNR, update magnitude, homeostatic contribution |
| `_diag_gated_stdp.py` | STDP vs PG magnitude, gated eligibility structure |
| `_diag_root_cause.py` | Eligibility sign analysis, weight floor, V(s) trajectory |

Po każdym fixie: uruchomić `_diag_learning.py` z 200 epizodami i sprawdzić metryki z tabeli §5.

---

## 8. DLACZEGO TE ZMIANY SKALUJĄ SIĘ DO AGI

1. **Usunięcie REINFORCE → pure STDP** skaluje bez problemu: STDP jest lokalna reguła O(active_synapses), nie wymaga backpropagation. REINFORCE wymaga ∇log π, który jest algorytmicznym artefaktem.

2. **NE multiplicative gain** jest biologicznie poprawny i skaluje do dowolnej sieci — modulator broadcast × local gain.

3. **InhibitoryPool** zapewnia E/I balance, który jest fundamentem KAŻDEJ stabilnej dynamiki neuronowej (Brunel 2000). Bez niego żaden system nie skaluje.

4. **Gate eligibility** to biologiczny mechanizm focused reinforcement, nie task-specific tuning.

5. **Normalizacja evidence** nie wpływa na architekturę — to kwestia odczytu sygnału.

Żadna z tych zmian nie jest "optymalizacją pod CartPole". To naprawa fundamentalnych bugów i usunięcie hack'ów ML, które blokują biologiczną dynamikę uczenia.

---

## 9. MOŻLIWE LIMITACJE PRĘDKOŚCI UCZENIA

Jeśli po FIX 1-7 uczenie jest wolniejsze niż expected (< 200 w 250 eps), to może wynikać z:

1. **gamma=0.95** limituje effective horizon do ~20 kroków. CartPole wymaga planowania 50-100 kroków. Podwyższenie do γ=0.99 wydłuża horizon do 100 kroków, ale spowalnia TD propagację (trade-off).

2. **SNN temporal integration** jest wolniejsze niż frame-by-frame DL — każda decyzja wymaga 25 substepów integracji membranowej. To jest ograniczenie biofizyczne (Wang 2002: cortical decisions 20-50ms). NIE PRZYSPIESZAĆ — to jest cena biologicznej wierności.

3. **Population noise** w małych populacjach (128 hidden, 32 per action) jest wyższa niż w biologicznym striatum (~10^6 MSN). To jest akceptowalne ograniczenie skali. NIE DODAWAĆ smoothing — noise jest biologicznie poprawna eksploracja.

4. **Sample efficiency** będzie gorsze niż DRL (DQN uczy się CartPole w ~100 eps). Ale DRL korzysta z replay buffer + target network + batch gradient. SNN uczy się online single-sample, jak biologiczny mózg. Biologiczna efektywność wymaga ~500+ prób dla prostych task'ów (Thorndike 1898: cats needed ~100+ trials for puzzle box). **250 epizodów jest ambitne ale osiągalne** dla poprawionego systemu.
