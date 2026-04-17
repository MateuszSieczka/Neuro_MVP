# Plan: SNN Brain for Embodied AGI

## TL;DR

Transform the current RL-centric SNN (Neuro_MVP_6) into a scalable, biologically-grounded artificial brain for embodied AGI. Migrate from NumPy to JAX for GPU+JIT (1M+ neurons). Replace the flat agent architecture with a hierarchical brain composed of differentiated regions (cortex, thalamus, BG, hippocampus, cerebellum). Add sensory pipelines (vision, auditory, touch), motor systems (movement + speech via phonemes), and cross-modal concept binding. Every component justified by papers — zero magic values, zero ML hacks.

---

## PHASE 0: CLEANUP & FOUNDATION

### 0.1 Remove dead code

- `core/pyramidal_neuron.py` — **DELETE entirely**. PyramidalLayer is never instantiated. Useful BAC-firing logic (Larkum 1999, Payeur 2021) will be re-integrated into the unified cortical neuron in Phase 1.
- `core/columnar.py` — **DELETE entirely**. Never activated (state_size < 16 for all tasks). Replace with `core/cortex.py` in Phase 2.
- `core/predictive_coding.py` — **DELETE**. Merge the Rao & Ballard relaxation loop into `core/error_neuron.py` (which is the superior implementation — dual-population, conductance-based, used by WorldModel).
- `core/competitive_layer.py` — **DELETE as standalone file**. Merge k-WTA inhibition into `core/cortex.py` (cortical column builder).
- Dead code within files:
  - `neuron.py`: Remove `_mem_decay`/`_mem_gain` (lines 194-195 — set but never read in forward()), `LIFLayer = AdExLayer` alias (line 507)
  - `config.py`: Remove `i_thresh` field (wrong formula, never used), `SynapseType` enum (never imported), fix `mg_concentration` bug (hardcoded in `nmda_mg_block()` instead of using field)
  - `synapse.py`: Remove unused `NeuronConfig` import
  - `simulation_context.py`: Remove unused `lru_cache` import
  - `basal_ganglia.py`: Remove `_STDP_LTD_LTP_RATIO` (declared, never used), `_msn_decay()` function (never called), `_t_since_pre/post` timing arrays (updated but never queried), `_homeo_counter` (incremented, never read)
  - `working_memory.py`: Remove `prediction_error` placeholder (initialized to ones, never updated)
  - `replay_buffer.py`: Remove `Experience.curiosity` (stored, never read), fix double-copy in `store()` (replace + **post_init** both copy)
  - `sequence_memory.py`: Remove `_step_count` (incremented, never read)
  - `network.py`: Fix `_concat_offsets` key direction bug (source,target vs target,source inconsistency)
  - `world_model.py`: Cap `error_history` to ring buffer (memory leak), remove `set_ne_level()` no-op
  - `astrocyte.py config`: Remove `atp_seizure_hill_n` and `atp_seizure_duration` (placeholders, never implemented)

### 0.2 Fix known bugs

- **`a_plus` double application** in `neuron.py`: Applied both in eligibility trace (L353/L360) AND in `update_weights()` (L475). Fix: remove from `update_weights()`, keep only in eligibility computation.
- **Jacobian missing ∂I_syn/∂V** in `neuron.py` L313-318: For conductance-based synapses, J(V) should include `-g_exc/C_m`. Fix: pass `g_exc_total` from `SynapticChannels.compute_current()` and add to Jacobian.
- **`mg_concentration` field ignored**: `SynapseConfig.nmda_mg_block()` hardcodes `1.0/3.57` instead of `self.mg_concentration/3.57`. Fix: use the config field.
- **World model τ_Ca inconsistency**: `world_model.py` L95 overrides `tau_ca=500` (10× faster than biological default 5000). Either justify or use default.

### 0.3 Resolve unit inconsistencies

- **Curiosity signal** in `world_model.py`: Mixes decoder_error (state units), encoder_error (spikes²/step), precision (dimensionless). Fix: normalize each component to [0,1] before combining. Reference: Barto et al. (2013) "Intrinsic motivation and reinforcement learning" — curiosity should be dimensionless information gain.
- **Mental rehearsal epistemic**: Same mix of spikes + variance + precision. Fix: express all components as information-theoretic quantities (nats or bits). PE → KL divergence, ambiguity → conditional entropy, precision → inverse Fisher information.
- **Attention gains**: Applied multiplicatively to currents without explicit units. Fix: make gains dimensionless [0.1, 2.0] and document clearly.

---

## PHASE 1: JAX MIGRATION & CORE ENGINE

### 1.1 JAX backend layer

Create `core/backend.py`:

- Abstract array operations behind thin wrapper
- `import jax.numpy as jnp` as primary, `numpy` as fallback
- Use `equinox` library for class-based JAX modules (jit-compatible)
- All neuron/synapse state becomes a JAX Pytree
- Forward pass: pure function `new_state, spikes = step(state, input, params)` — jit-compilable
- Use `jax.lax.scan` for temporal simulation (auto-unroll substeps)
- Use `jax.vmap` for batch processing across independent brain regions

### 1.2 Sparse connectivity engine

Create `core/sparse.py`:

- **CSR sparse matrix** representation for synaptic weights (via `jax.experimental.sparse`)
- Connection probability masks: `p_connect(distance) = p_local × exp(-d²/2σ²)` (Hellwig 2000 — distance-dependent cortical connectivity)
- Block-sparse for local connectivity within cortical columns
- Random long-range sparse connections (Markov et al. 2014 — cortical interarea connectivity)
- Sparse matmul: `I_syn = sparse_w @ pre_spikes` — O(nnz) instead of O(N²)
- **Structural plasticity**: Functions for synaptogenesis (add connections where correlated activity has no synapse) and pruning (remove synapses below weight threshold). Reference: Butz & van Ooyen (2013) "A simple rule for dendritic spine and axonal bouton formation"
- Target: ~5-15% connectivity density (Braitenberg & Schüz 1998 — cortical statistics)

### 1.3 Unified neuron model

Refactor `core/neuron.py` to support two modes in a single jit-compiled function:

**Mode A — LIF+SFA** (default, 90% of neurons):

```
V_{t+1} = V_t × decay + (I_syn - w_adapt) × gain
w_{t+1} = w_t × w_decay + b × spike_t
spike = V > V_thresh
```

- decay = exp(-dt/τ_m), gain = (1 - decay) / g_L
- Reference: Gerstner & Kistler (2002), Brette & Gerstner (2005) limiting case
- ~3× faster than AdEx (no exp computation per neuron per step)
- SFA captures working memory dynamics (Benda & Herz 2003)

**Mode B — AdEx** (for PFC attractor / thalamic burst):

- Keep current implementation, port to JAX
- Use only where burst detection or precise adaptation dynamics are critical
- Reference: Brette & Gerstner (2005)

**Common features:**

- Refractory period (absolute + relative via adaptation)
- Spike traces for STDP (exponential decay)
- Conductance-based synaptic input via SynapticChannels
- Per-neuron configurable: `mode = 'lif' | 'adex'` flag
- All state as flat JAX arrays (no Python objects per neuron)

### 1.4 Event-driven optimization (for 1M+ scale)

At 1M+ neurons, even O(N) per timestep is expensive for non-spiking neurons. Implement **hybrid clock-driven / event-driven**:

- **Clock-driven** for active populations (recently received input or near threshold)
- **Inactive populations**: Skip integration, maintain at resting potential
- **Activation queue**: When a presynaptic population spikes, mark postsynaptic population as "active" for τ_m timesteps
- Reference: Brette et al. (2007) "Simulation of networks of spiking neurons" — the standard approach for large-scale SNN
- JAX implementation: use masks + conditional computation via `jax.lax.cond`

---

## PHASE 2: BRAIN REGIONS ARCHITECTURE

### 2.1 CorticalArea module

Create `core/cortex.py` — generic cortical area factory:

Each cortical area has 3 functional layers (simplified from 6 biological):

- **L2/3** (superficial): Predictive coding — generates predictions, computes PE. Uses ErrorNeuronLayer dual-population model (state + error neurons). Reference: Rao & Ballard (1999), Bastos et al. (2012) "Canonical microcircuits for predictive coding"
- **L4** (granular): Input layer — receives feedforward thalamocortical input. k-WTA competitive inhibition for sparse coding. Reference: Douglas & Martin (2004) "Neuronal circuits of the neocortex"
- **L5/6** (deep): Output layer — sends feedback/efferent projections. Generates top-down predictions. Adaptive threshold for attention gating.

**Connectivity pattern** (Bastos et al. 2012 canonical microcircuit):

- Feedforward: L4 → L2/3 → L5/6
- Feedback: L5/6 → L2/3 (top-down prediction)
- Lateral: Within-layer recurrent (attractor dynamics)
- Inhibitory: Each layer has a local InhibitoryPool (FS interneurons)

**Parameters per area** (configurable):

- `n_neurons_per_layer`: list[int] — e.g., [256, 512, 256] for L2/3, L4, L5/6
- `connectivity_density`: float — default 0.1 (10%)
- `neuron_mode`: 'lif' (default) or 'adex' (for PFC)
- `has_predictive_coding`: bool — whether L2/3 does PC
- `receptive_field_size`: Optional[int] — for sensory areas with topographic maps

**Interarea connections** (Markov et al. 2014 — SLN hierarchy):

- Feedforward: L2/3 of lower area → L4 of higher area
- Feedback: L5/6 of higher area → L2/3 of lower area (targets error neurons)
- Each connection has: weight, delay (axonal conduction), connection probability

### 2.2 Thalamus module

Create `core/thalamus.py`:

- **Thalamic relay neurons**: AdEx in burst mode (IB type: a=2, b=60, τ_w=20)
  - Two states: burst (low DA) and tonic (high DA) — Sherman & Guillery (2002)
  - Burst mode gates novel/salient inputs → cortex
  - Tonic mode passes ongoing sensory stream → cortex
- **Reticular nucleus (TRN)**: Inhibitory shell around thalamus
  - Receives collaterals from both cortex→thalamus and thalamus→cortex
  - Provides lateral inhibition between thalamic nuclei
  - Implements attentional selection (Crick 1984 "Function of the thalamic reticular complex")
- **Nuclei** (configurable):
  - LGN (vision) → V1
  - MGN (audition) → A1
  - VPL/VPM (somatosensory) → S1
  - Pulvinar (association/attention) → higher cortical areas
  - MD (mediodorsal) → PFC
- Reference: Sherman & Guillery (2006) "Exploring the Thalamus and Its Role in Cortical Function"

### 2.3 Cerebellum module

Create `core/cerebellum.py`:

- **Granule cell layer**: Massive expansion (N_granule ≈ 4 × N_mossy) with sparse random projections — like DG in hippocampus. Reference: Marr (1969) "A theory of cerebellar cortex"
- **Purkinje cells**: Main output — learn via climbing fiber error signals. LTD at parallel fiber→Purkinje synapses when climbing fiber is active. Reference: Albus (1971), Ito (2006) "Cerebellar circuitry as a neuronal machine"
- **Deep cerebellar nuclei**: Output pathway, inhibited by Purkinje cells
- **Function**: Forward model — learns to predict sensory consequences of actions:
  - Input: efference copy from motor cortex + current sensory state
  - Output: predicted next sensory state
  - Error: actual vs predicted sensory state (from climbing fibers / inferior olive)
  - Reference: Wolpert et al. (1998) "Internal models in the cerebellum"
- This replaces the current `world_model.py` for motor predictions (world_model stays for higher-level cognitive predictions)

### 2.4 BrainGraph orchestrator

Refactor `core/network.py` → `core/brain_graph.py`:

- Replace flat layer graph with **hierarchical region graph**
- Each region is a `CorticalArea`, `ThalamusNucleus`, `CerebellumCircuit`, `BasalGangliaCircuit`, or `HippocampalFormation`
- **Inter-region connections**: Sparse, delayed (axonal conduction time 1-10 ms per brain region distance)
  - Reference: Swadlow (2000) — axonal conduction velocities 1-10 m/s
  - With brain-scale distances of 1-10 cm: delays of 1-10 ms
- **Global oscillator bus**: Each region subscribes to theta/gamma phase
  - But can have local oscillator phase offsets (traveling waves)
  - Reference: Muller et al. (2018) "Cortical travelling waves"
- **Neuromodulator distribution**: Global + regional modulation (keep current architecture but expand to per-region NE/ACh)
- **Step function**: Topological order respecting delays, jit-compiled

---

## PHASE 3: SENSORY SYSTEMS

### 3.1 Vision pipeline

Create `sensory/vision.py`:

**Architecture** (Hybrid Gabor V1 + spiking higher):

1. **Retina** (fixed, not learned):
   - Input: RGB image (64×64 or 128×128 from Unity camera)
   - Center-surround: Difference of Gaussians (ON-center/OFF-center ganglion cells)
   - Temporal change detection: Frame differencing → event-like spikes
   - Reference: Rodieck (1998) "The First Steps in Seeing"
   - Implementation: Standard convolution on GPU (JAX conv2d)

2. **LGN** (thalamic relay):
   - Burst/tonic mode for novel vs familiar visual input
   - Attentional gating via TRN feedback

3. **V1** (fixed Gabor filters, not learned):
   - 8 orientations × 4 spatial frequencies × ON/OFF = 64 feature maps
   - Gabor filter bank: `g(x,y) = exp(-(x'²+γ²y'²)/2σ²) × cos(2πx'/λ + φ)`
   - Reference: Hubel & Wiesel (1962), Olshausen & Field (1996)
   - Implementation: Depthwise convolution on GPU (fast, parallel)
   - Output: Sparse spike patterns (threshold + WTA competition within orientation columns)

4. **V2** (learned via STDP — CorticalArea):
   - Combines V1 features into texture/contour elements
   - Reference: Freeman et al. (2013) — V2 texture selectivity
   - n_neurons: ~2000 (configurable)

5. **V4/IT** (learned — CorticalArea):
   - Object-level representations
   - Invariant to position/size via progressive pooling
   - Reference: DiCarlo et al. (2012) "How does the brain solve visual object recognition?"
   - n_neurons: ~1000 (configurable)
   - Outputs: Object-level sparse codes that feed into concept binding

**Optional semantic shortcut** (removable):

- DINOv2 or CLIP features → population encoding → inject into V4/IT as additional input
- For bootstrapping early training when visual cortex hasn't learned enough
- Reference: Conceptually similar to "innate knowledge" / evolutionary priors
- Can be gradually faded out as SNN visual hierarchy matures

### 3.2 Auditory pipeline

Create `sensory/auditory.py`:

1. **Cochlea** (fixed, not learned):
   - Mel filterbank (64 frequency bands) applied to audio waveform
   - Half-wave rectification (inner hair cell model)
   - Reference: Lyon (2017) "Human and Machine Hearing"
   - Output: 64-channel spectrogram at ~1ms resolution

2. **MGN** (thalamic relay):
   - Tonotopic organization preserved
   - Burst mode for novel sounds

3. **A1** (primary auditory cortex — CorticalArea):
   - Tonotopic map (frequency → spatial position)
   - Temporal pattern detection (onset, offset, frequency modulation)
   - Reference: Kaas & Hackett (2000)
   - n_neurons: ~1000

4. **Belt/Parabelt** (secondary auditory — CorticalArea):
   - Phoneme-level representations
   - Temporal sequence learning (sequence_memory integration)
   - Reference: Rauschecker & Scott (2009) "Maps and streams in the auditory cortex"
   - n_neurons: ~500

5. **STS** (superior temporal sulcus — CorticalArea):
   - Audio-visual integration
   - Voice/face binding
   - Reference: Beauchamp et al. (2004)

### 3.3 Somatosensory pipeline

Create `sensory/somatosensory.py`:

1. **Mechanoreceptors** (fixed encoding):
   - Pressure/force → population coded spikes
   - Proprioception: joint angles + velocities → population coded
   - Reference: Johansson & Flanagan (2009)

2. **S1** (primary somatosensory — CorticalArea):
   - Somatotopic map (body part → spatial position)
   - Texture, pressure, proprioception integration

3. **S2** (secondary — CorticalArea):
   - Higher-order tactile processing
   - Object recognition by touch

### 3.4 Interoception

Create `sensory/interoception.py`:

- Internal state variables: energy level (from astrocyte ATP), "pain" (prediction error magnitude), arousal (NE level)
- Population coded → insular cortex area
- Reference: Craig (2009) "How do you feel — now?"
- Drives homeostatic behavior (seek food when energy low, rest when fatigued)

---

## PHASE 4: MOTOR SYSTEMS + LANGUAGE

### 4.1 Motor cortex

Create `motor/motor_cortex.py`:

1. **Premotor cortex** (CorticalArea):
   - Action plan level: high-level goals → sequences of motor primitives
   - Receives from PFC (goals), BG (selected action)
   - Outputs to M1
   - Reference: Rizzolatti & Luppino (2001) "The cortical motor system"

2. **Primary motor cortex M1** (CorticalArea):
   - Motor primitive execution: population-coded movement commands
   - Output: joint angle targets / velocities / torques (configurable)
   - Somatotopic map (body part → spatial position in M1)
   - Reference: Georgopoulos et al. (1986) — population vector coding

3. **Body interface** (`body_interface.py`):
   - Abstract interface: `send_motor_command(joint_targets)`, `receive_sensory(modality)`
   - Unity adapter: gRPC/WebSocket to Unity simulation
   - MuJoCo adapter: Direct Python bindings
   - Real robot adapter: ROS2 messages
   - Reference: Pfeifer & Bongard (2006) "How the Body Shapes the Way We Think"

### 4.2 Phoneme / speech system

Create `motor/speech.py`:

1. **Phoneme inventory** (configurable, default IPA subset):
   - ~40 phonemes for English (expandable)
   - Each phoneme = activation pattern in speech motor cortex
   - Reference: Levelt (1989) "Speaking: From Intention to Articulation"

2. **Broca's area** (CorticalArea):
   - Phoneme sequence planning
   - Receives from: PFC (intended meaning), Wernicke's (comprehended speech for repetition)
   - Outputs: Ordered phoneme activations
   - Syllabification and prosody
   - Reference: Flinker et al. (2015) "Redefining the role of Broca's area in speech"

3. **Speech motor cortex** (part of M1):
   - Phoneme activation → articulatory motor commands
   - Output: phoneme ID + onset timing → external TTS engine
   - Reference: Guenther (2006) "Cortical interactions underlying the production of speech sounds"

4. **Babbling loop** (learning mechanism):
   - **Phase 1 — Vocal babbling**: Random phoneme activation → TTS → audio input → A1 processing
     - Agent hears its own "voice" through the auditory pipeline
     - Builds auditory-motor mapping through STDP
     - Reference: Kuhl (2004) "Early language acquisition: cracking the speech code"
   - **Phase 2 — Imitation**: Hear external word → activate Wernicke's → attempt reproduction via Broca's → compare heard vs intended
     - Error signal: prediction error between heard phoneme sequence and target
     - Drives learning in Broca's→M1 pathway
     - Reference: Rizzolatti & Arbib (1998) — mirror neuron hypothesis for speech
   - **Phase 3 — Naming**: See object → retrieve associated word → produce speech
     - Cross-modal binding: visual concept → semantic → phonological → motor
     - Reference: Levelt et al. (1999) "A theory of lexical access in speech production"

### 4.3 Wernicke's area

Part of auditory hierarchy (STS + posterior temporal):

- Phoneme sequences → word recognition
- Maps to semantic concepts
- Reference: Hickok & Poeppel (2007) "The cortical organization of speech processing" (dual-stream model)

---

## PHASE 5: MEMORY SYSTEMS

### 5.1 Working memory (PFC refactoring)

Refactor `core/working_memory.py`:

- Keep current attractor + dual ACh/DA gating (well-implemented)
- Add: **Multiple WM slots** — not one monolithic WM, but N independent attractor networks (each with ~64-128 neurons)
  - Capacity ~4±1 chunks (Cowan 2001)
  - Each slot maintained by persistent activity + NMDA currents
  - Slots compete via lateral inhibition
  - Reference: Bays & Husain (2008) "Dynamic shifts of limited working memory resources"
- Add: **Goal register** — special WM slot for current goal, gated by dorsolateral PFC
  - Maintained until goal is achieved or explicitly overwritten
  - Reference: Miller & Cohen (2001) "An integrative theory of prefrontal cortex function"
- Switch to AdEx neurons (needed for persistent activity / attractor stability)

### 5.2 Episodic memory (Hippocampus expansion)

Refactor `core/episodic_memory.py` → `core/hippocampus.py`:

- Keep current DG sparse coding + CA3 pattern completion
- Add: **CA1 temporal binding** — temporal context vectors that allow time-stamping memories
  - Reference: Howard & Kahana (2002) "A distributed representation of temporal context"
  - Implementation: Slowly drifting context vector that provides temporal tagging
- Add: **Place cells / Grid cells** (for spatial navigation in embodied env)
  - Grid cells: periodic spatial firing patterns (Hafting et al. 2005)
  - Place cells: location-specific firing (O'Keefe & Nadel 1978)
  - Implementation: Population of neurons with grid-like response functions + Hebbian learning
  - These form the spatial scaffold for episodic memories
  - Reference: Moser et al. (2008) "Place cells, grid cells, and the brain's spatial representation system"
- Add: **Sharp-wave ripple (SWR) generation** — currently SWRs are externally triggered; make them endogenous
  - CA3 recurrent excitation builds up → CA1 ripple → cortical reactivation
  - Reference: Buzsáki (2015) "Hippocampal sharp wave-ripple"
- Add: **Consolidation pathway**: Hippocampus → cortex (slow, during sleep)
  - Repeated replay gradually transfers episodic → semantic memory
  - Reference: McClelland et al. (1995) — complementary learning systems theory

### 5.3 Semantic memory (Temporal cortex)

New: Represented as stable weight patterns in association cortex

- Formed through repeated activation (consolidation from episodic)
- Cross-modal: concept "apple" = visual features + taste + word sound + motor grasp
- Implemented via convergence zone architecture
- Reference: Damasio (1989) "Time-locked multiregional retroactivation" — convergence zones for concepts
- Not a separate module — emerges from connectivity between sensory areas + PFC + hippocampus

### 5.4 Procedural memory (BG habits)

Already partially implemented in `basal_ganglia.py`:

- D1/D2 STDP-based learning → habit formation
- Keep but streamline: Remove RL-specific scaffolding, make generic action selection
- Add: **Chunking** — sequences of actions that become automatic
  - Fast-learning BG pathway for initial learning → slow transfer to cortical habits
  - Reference: Graybiel (2008) "Habits, rituals, and the evaluative brain"

### 5.5 Sleep consolidation (Replay buffer refactoring)

Refactor `core/replay_buffer.py`:

- Keep SWS + REM two-phase design
- Fix ML hacks:
  - Replace `max(abs(rpe), 0.1)` floor with astrocyte-derived metabolic modulation (ATP level → learning gate)
  - Replace fixed `m_t = 0.5` for REM with serotonin-modulated learning rate
  - Replace `2.5` GABA surge with astrocyte-calibrated inhibitory gain
- Add: **Dreaming** — REM forward replay that generates novel combinations
  - Not just replaying stored experiences, but recombining them
  - Reference: Hobson (2009) "REM sleep and dreaming"
  - Implementation: World model generates imagined trajectories → train on them
- Wire astrocyte ATP to sleep trigger (currently dead code — Phase 0 fix)

---

## PHASE 6: LEARNING & PLASTICITY

### 6.1 Synaptic plasticity (clean up existing)

- **STDP**: Keep Bi & Poo (2001) kernel. Fix `a_plus` double application.
- **Three-factor**: Keep modulation × eligibility × error. Clean up unit mismatches.
- **Homeostatic scaling**: Keep Turrigiano (2008). Fix hardcoded `0.05` rate — use config.
- **BCM threshold**: Keep. Already well-implemented.
- All learning rules ported to JAX (differentiable for debugging, but not used for backprop)

### 6.2 Structural plasticity (NEW)

Add to `core/sparse.py`:

- **Synaptogenesis**: Correlated pre/post activity without direct connection → add synapse
  - Rule: If pre and post spike within 20ms AND no synapse exists AND random test passes → create synapse with small initial weight
  - Reference: Butz & van Ooyen (2013)
  - Rate: ~0.1% of potential connections checked per second (stochastic)
- **Pruning**: Synapses with weight below threshold for extended period → remove
  - Threshold: 1% of mean weight in population
  - Holdout period: ~10 minutes (biological: days-weeks, compressed)
  - Reference: Chechik et al. (1998) "Synaptic pruning in development"
- This is how the brain GROWS new circuits for new concepts

### 6.3 Neuromodulation (keep + expand)

Keep current 4-channel system (DA, ACh, NE, 5-HT). Modifications:

- Wire astrocytes to all layers (currently dead in production)
- Add **regional neuromodulation** — currently global; make per-region:
  - VTA → dorsal striatum (phasic DA for action learning)
  - VTA → PFC (tonic DA for working memory gating)
  - Locus coeruleus → all cortex (NE for arousal/attention)
  - Basal forebrain → cortex (ACh for learning mode vs recall mode)
  - Dorsal raphe → everywhere (5-HT for temporal horizon + mood)
  - Reference: Doya (2002) "Metalearning and neuromodulation"

### 6.4 Continuous learning (meta-learning via neuromodulation)

The brain must never stop learning, but must also not catastrophically forget.

- **Consolidation**: Sleep replays + hippocampal→cortical transfer
- **Protection of old memories**: Elastic weight consolidation-like mechanism, but biological:
  - Synaptic tagging and capture (Frey & Morris 1997): Strong synaptic events create "tags" that protect potentiated synapses
  - Implementation: Per-synapse "consolidation" flag (from replay count in episodic memory), consolidated synapses have slower learning rate
  - Reference: Redondo & Morris (2011) "Making memories last"
- **Novelty-driven plasticity**: NE/ACh increase plasticity for novel stimuli, decrease for familiar
  - Already partially implemented via neuromodulator system
  - Make more explicit: NE high → all STDP traces have shorter τ (faster learning)

---

## PHASE 7: CROSS-MODAL BINDING & CONCEPTS

### 7.1 Temporal binding via gamma synchronization

- Neurons representing features of same object fire in same gamma cycle
- Binding by synchrony (Engel & Singer 2001, Gray 1994)
- Implementation: Oscillator phase resets group co-active neurons
- Already partially implemented via theta-gamma oscillator; extend to per-region gamma
- Reference: Fries (2005) "A mechanism for cognitive dynamics: neuronal communication through neuronal coherence"

### 7.2 Convergence zones

- Create association areas where multiple sensory streams converge
- **Temporal pole**: Auditory + visual → audiovisual concepts
- **Angular gyrus**: Visual + auditory + somatosensory → rich multi-modal concepts
- **PFC**: All modalities + goals + memory → executive representations
- Implementation: CorticalAreas that receive projections from multiple sensory hierarchies
- Learning: Hebbian — co-activated features across modalities strengthen connections
- Reference: Damasio (1989) convergence zones, Patterson et al. (2007) "Where do you know what you know?"

### 7.3 Concept formation

A "concept" is a stable attractor in association cortex activated by any associated modality:

- See apple → visual pathway → IT features → apple concept activation
- Hear "apple" → auditory pathway → word recognition → apple concept activation
- Feel apple → tactile pathway → shape/texture → apple concept activation
- All three activate the SAME attractor population in convergence zone
- Reference: Binder et al. (2009) "Where is the semantic system?"

### 7.4 Inner speech / thought

"Thinking" = sequential activation of concept attractors without external input:

- PFC goal representation → primes relevant concepts in association cortex
- Hippocampal replay can trigger chains of associated concepts
- Working memory maintains current "thought" while next is being activated
- Reference: Fodor (1975) "Language of Thought", Carruthers (2002) "The cognitive functions of language"
- Implementation: PFC → BG action selection operates on internal "actions" (concept retrieval, WM slot manipulation) in addition to external motor actions
- **Key insight**: Same BG selection mechanism that chooses physical actions also chooses "mental actions" — which concept to attend to, what to retrieve from memory, what to say
  - Reference: Frank & Badre (2012) "Mechanisms of hierarchical reinforcement learning in corticostriatal circuits"

---

## PHASE 8: INTEGRATION — THE BRAIN CLASS

### 8.1 Brain class

Create `brain.py` — top-level orchestrator:

```
Brain
├── sensory/
│   ├── vision: RetinalProcessor → LGN → V1 → V2 → V4/IT
│   ├── auditory: Cochlea → MGN → A1 → Belt → STS
│   ├── somatosensory: Mechanoreceptors → VPL → S1 → S2
│   └── interoception: InternalState → Insula
├── cortex/
│   ├── temporal_association: V4/IT + STS → concepts
│   ├── parietal: S2 + V2 → spatial processing
│   ├── prefrontal: dlPFC (goals) + vmPFC (values) + Broca's
│   └── motor: Premotor → M1 → body_interface
├── subcortical/
│   ├── thalamus: LGN, MGN, VPL, Pulvinar, MD
│   ├── basal_ganglia: D1/D2 pathways (action selection)
│   ├── hippocampus: DG → CA3 → CA1 (memory)
│   ├── cerebellum: Forward models
│   ├── vta: Dopamine (RPE)
│   └── amygdala: (future — valence tagging)
├── neuromodulation/
│   ├── dopamine: VTA → BG, PFC
│   ├── noradrenaline: LC → all cortex
│   ├── acetylcholine: BF → cortex
│   └── serotonin: DR → everywhere
├── oscillator: Global theta-gamma + per-region phase
├── astrocyte_field: Per-region metabolic regulation
└── body_interface: Motor output + sensory input
```

### 8.2 Main loop (replaces SNNAgent)

```
every 1ms (dt):
  1. Receive sensory input from body_interface
  2. Encode through sensory pipelines (retina, cochlea, etc.)
  3. Thalamic relay (burst/tonic gating)
  4. Cortical processing (all areas in parallel, respect delays)
  5. PFC goal maintenance + BG action selection
  6. Motor output → body_interface
  7. Neuromodulator update (global)
  8. Astrocyte update (per-region)
  9. Oscillator tick
  10. STDP + homeostatic plasticity

every sleep_cycle:
  SWS: reverse replay + hippocampal consolidation
  REM: forward replay + dreaming + sequence learning
```

### 8.3 Body interface protocol

Create `body_interface.py`:

```python
class BodyInterface(ABC):
    def receive_vision(self) -> jnp.ndarray:  # RGB image
    def receive_audio(self) -> jnp.ndarray:   # waveform
    def receive_touch(self) -> jnp.ndarray:   # pressure map
    def receive_proprioception(self) -> jnp.ndarray:  # joint angles
    def send_motor(self, joint_targets: jnp.ndarray) -> None
    def send_speech(self, phoneme_ids: jnp.ndarray) -> None
```

Implementations:

- `UnityInterface(BodyInterface)` — gRPC to Unity
- `MuJoCoInterface(BodyInterface)` — direct Python bindings
- `ROS2Interface(BodyInterface)` — ROS2 topics

---

## RELEVANT FILES

### Keep & Modify

- `core/config.py` — Refactor into per-region configs, fix mg_concentration bug, remove dead fields
- `core/neuron.py` — Add LIF+SFA mode, port to JAX, fix a_plus bug, fix Jacobian
- `core/synapse.py` — Port to JAX, add sparse matmul path
- `core/neuromodulator.py` — Port to JAX, add per-region modulation
- `core/receptor.py` — Port to JAX (minimal changes)
- `core/vta.py` — Port to JAX (minimal changes)
- `core/oscillator.py` — Port to JAX, add multi-regional phase
- `core/astrocyte.py` — Port to JAX, wire to production
- `core/free_energy.py` — Port to JAX (minimal changes)
- `core/simulation_context.py` — Port to JAX
- `core/working_memory.py` — Multi-slot WM, AdEx mode
- `core/episodic_memory.py` → rename to `core/hippocampus.py`, expand
- `core/basal_ganglia.py` — Streamline, remove RL scaffolding
- `core/sequence_memory.py` — Port to JAX (minimal changes)
- `core/replay_buffer.py` — Fix ML hacks, add dreaming
- `core/attention.py` — Expand to thalamic gating
- `core/network.py` → refactor into `core/brain_graph.py`
- `core/error_neuron.py` — Port to JAX, absorb predictive_coding logic
- `core/interneuron.py` — Port to JAX, merge into cortex module
- `core/spike_encoder.py` — Port to JAX, add modality-specific encoders
- `arena/core.py` — Keep interface ABCs, remove Trainer (replaced by brain loop)

### Delete

- `core/pyramidal_neuron.py` — Dead, merge BAC firing into cortex
- `core/columnar.py` — Dead, replace with cortex.py
- `core/predictive_coding.py` — Merge into error_neuron.py
- `core/competitive_layer.py` — Merge into cortex.py
- `arena/snn_agent.py` — Replace with brain.py
- `arena/agent_factory.py` — Replace with brain configuration
- `arena/benchmark.py` — Replace with embodied evaluation
- `arena/task_config.py` — Replace with embodied task configs
- `arena/environments.py` — Replace with body_interface
- `arena/gym_env.py` — Replace with body_interface adapters

### Create New

- `core/backend.py` — JAX backend abstraction
- `core/sparse.py` — Sparse connectivity engine
- `core/cortex.py` — Generic cortical area factory
- `core/thalamus.py` — Thalamic relay module
- `core/cerebellum.py` — Forward model circuit
- `core/hippocampus.py` — Expanded hippocampal formation (from episodic_memory)
- `core/brain_graph.py` — Hierarchical brain region orchestrator
- `sensory/vision.py` — Visual processing pipeline
- `sensory/auditory.py` — Auditory processing pipeline
- `sensory/somatosensory.py` — Touch + proprioception
- `sensory/interoception.py` — Internal state sensing
- `motor/motor_cortex.py` — Motor output system
- `motor/speech.py` — Phoneme system + babbling
- `brain.py` — Top-level Brain class
- `body_interface.py` — Abstract embodiment interface

---

## VERIFICATION

1. **Unit tests**: Each brain region independently testable — fixed input → expected firing pattern
2. **Integration test — Babbling**: Brain produces random phonemes → hears own voice → reduces error over time (auditory-motor mapping forms)
3. **Integration test — Object naming**: Present object + play word → brain associates → later, present object → brain produces word
4. **Integration test — Navigation**: Simple grid world → brain learns to navigate to goal using hippocampal place cells + BG action selection
5. **Benchmark — Spike statistics**: Verify biologically plausible firing rates (1-20 Hz cortical, 50-200 Hz interneurons) across all regions
6. **Benchmark — Memory**: Store N episodes → recall accuracy > 80% for recent, graceful degradation for old
7. **Benchmark — Scaling**: 1M neurons, <100ms wall-clock per brain step on single GPU (JAX jit)
8. **Benchmark — Continuous learning**: Train task A → train task B → verify task A performance retained above 70%
9. **Diagnostic — Weight statistics**: No runaway weights, mean weight per region stays within 2× initial over 10K steps
10. **Diagnostic — ATP dynamics**: Astrocyte ATP depletes during sustained activity, recovers during sleep, triggers sleep when critical

---

## DECISIONS

1. **AdEx vs LIF**: LIF+SFA for 90% of neurons (3× cheaper), AdEx only for PFC attractor and thalamic burst mode. Justified: Benda & Herz (2003) show SFA captures essential adaptation dynamics.
2. **Vision**: Hybrid Gabor V1 (fixed, not learned) + spiking V2+ (STDP learned). Most future-proof — scales to real cameras, allows incremental learning, biologically grounded (Hubel & Wiesel 1962). Optional DINOv2 shortcut for bootstrapping.
3. **Speech**: Phoneme inventory in SNN → external TTS. Learning via babbling loop. Most biologically accurate approach (Kuhl 2004).
4. **JAX**: Full migration from NumPy. Required for 1M+ neuron scale. Equinox for class-based modules.
5. **Sparse connectivity**: 5-15% density (Braitenberg & Schüz 1998). CSR format via jax.experimental.sparse. Critical for O(N) vs O(N²) scaling.
6. **No backprop anywhere**: All learning is local (STDP, Hebbian, three-factor). Biologically grounded.
7. **Body interface**: Abstract ABC with Unity/MuJoCo/ROS2 adapters. Start with Unity, migrate to real robot later.
8. **Sleep**: ATP-driven (endogenous), not episode-boundary triggered. Finally wire dormant astrocyte pathway.
9. **Concepts**: Emerge from cross-modal Hebbian binding in convergence zones, not explicitly programmed.
10. **Thought**: Same BG selection mechanism for internal "mental actions" as for external motor actions (Frank & Badre 2012).

---

## FURTHER CONSIDERATIONS

1. **Amygdala**: Valence tagging (fear/reward) for emotional memory. Not included in initial plan but natural extension. Would add prioritization to episodic storage and bias BG selection. Reference: LeDoux (2000).
2. **Mirror neurons**: Premotor neurons that fire both during action execution and observation. Important for imitation learning and empathy. Could emerge naturally from sensory-motor predictive coding. Reference: Rizzolatti & Craighero (2004).
3. **Attention schema**: Explicit self-model for attention allocation. The brain modeling its own attention process. Would support metacognition. Reference: Graziano (2013) "Consciousness and the Social Brain".

---

## IMPLEMENTATION ORDER

The phases are ordered by dependency:

```
Phase 0 (Cleanup) ─────────┐
                            ▼
Phase 1 (JAX + Core) ──────┤  ← Foundation, everything depends on this
                            ▼
Phase 2 (Brain Regions) ────┤  ← depends on Phase 1
      ┌─────────┬──────────┤
      ▼         ▼          ▼
Phase 3      Phase 4    Phase 5    ← Sensory, Motor, Memory (parallel)
(Sensory)    (Motor)    (Memory)
      └─────────┴──────────┘
                ▼
Phase 6 (Learning) ────────── ← depends on all above
                ▼
Phase 7 (Cross-modal) ─────── ← depends on sensory + memory
                ▼
Phase 8 (Integration) ──────── ← depends on everything
```

Phases 3, 4, 5 can be developed in parallel once Phase 2 is complete.
