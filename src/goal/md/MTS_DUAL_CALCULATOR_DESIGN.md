# Multiple-Time-Step MD with Dual Calculator Fine-Tuning

## Architecture Design

### Overview

Integration of i-pi's accurate MTS physics with goal.md's dual-calculator system enables:

1. **High-accuracy base calculations** (QM/expensive ML)
2. **Fast ML model training** on-the-fly from base calculator
3. **Intelligent switching** between base and trained calculator
4. **Config-driven orchestration** of multi-calculator workflows

### Key Components

```
goal.md/
├── mts/                          # NEW: Multiple-time-step module
│   ├── __init__.py
│   ├── integrators/              # i-pi integrator adapters
│   │   ├── __init__.py
│   │   ├── base_integrator.py   # Wrapper around i-pi integrators
│   │   ├── mts_integrator.py    # MTS-specific integration
│   │   └── dynamics_converter.py # Convert i-pi↔ASE
│   │
│   ├── dual_calculator/          # Dual-calculator system
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract dual-calculator
│   │   ├── qm_ml_learner.py    # QM teacher + ML student
│   │   ├── calculator_switch.py # Switching logic & strategies
│   │   └── training.py          # Online ML training
│   │
│   ├── ensemble_managers/        # Manage dual calculators
│   │   ├── __init__.py
│   │   └── two_level.py         # 2-calculator orchestration
│   │
│   └── config_parsers/          # Hydra config support
│       ├── __init__.py
│       └── mts_config.py
│
├── adapters/                     # Enhanced adapters
│   ├── ipi_converter.py         # ASE Atoms ↔ i-pi structures
│   └── existing converters...
```

### Dual-Calculator System

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ MTS MD Engine (i-pi based)                                  │
└──────────────┬──────────────────────────────────────────────┘
               │
        ┌──────┴─────┐
        │             │
        ▼             ▼
┌─────────────┐  ┌─────────────────────┐
│ Base Level  │  │ ML Fine-tuning Level│
├─────────────┤  ├─────────────────────┤
│ QM/Exp.     │  │ ML Model being      │
│ Calculator  │  │ trained on base     │
│             │  │                     │
│ (always     │  │ (adaptive training) │
│  running)   │  │                     │
└─────────────┘  └─────────────────────┘
        │             │
        └──────┬──────┘
               │
    ┌──────────▼──────────┐
    │ Calculator Switcher │
    │ (accuracy monitor)  │
    └─────────────────────┘
               │
        ┌──────▼───────┐
        │ Logic: when  │
        │ ML accurate? │
        │ Switch base? │
        │ Retrain?     │
        └──────────────┘
```

#### States

```
State Transitions:

INITIALIZE
    ↓
    │ Setup QM calculator, prepare ML model
    ▼
QM_RUNNING (ML training in parallel)
    ↓
    │ Every N steps: evaluate ML accuracy
    ├─→ accuracy < threshold  ──→ CONTINUE (more training)
    ├─→ accuracy ≥ threshold  ──→ SWITCH_READY
    └─→ error detected        ──→ FALLBACK_QM
    ▼
SWITCH_READY (ML takes over)
    ↓
ML_RUNNING (QM standby)
    ├─→ periodically: validation checks
    ├─→ accuracy drift detected  ──→ REVERT_QM
    └─→ update/retrain triggered  ──→ RETRAIN
    ▼
RETRAIN (parallel update)
    ├─→ update complete ──→ ML_RUNNING
    └─→ too much drift  ──→ FALLBACK_QM
```

### Configuration Schema

```yaml
# configs/md/mts/dual_calculator_qm_ml.yaml
mts:
  enabled: true
  backend: ipi  # i-pi integration engine

  base_calculator:
    type: goal_model  # Or orca, cp2k, etc.
    checkpoint: path/to/base_model.ckpt
    update_frequency: 1  # steps between force updates

  ml_calculator:
    type: goal_model
    checkpoint: path/to/initial_ml_model.ckpt
    online_training: true

  switching_strategy:
    type: accuracy_based  # or: time_based, energy_based, hybrid
    parameters:
      force_rmse_threshold: 0.01  # eV/Å
      energy_mae_threshold: 0.001  # eV/atom
      evaluation_interval: 100  # steps
      min_training_steps: 500

  training:
    method: online_gradient  # or: batch_replay, experience_replay
    batch_size: 32
    learning_rate: 0.001
    loss_function: mse_weighted  # force-weighted loss

  monitoring:
    log_interval: 10
    save_checkpoint_interval: 100
    metrics:
      - force_rmse
      - energy_mae
      - model_uncertainty
      - switching_events

  fallback:
    mode: automatic  # or: manual, rl_based
    triggers:
      - training_loss_increase
      - prediction_uncertainty_high
      - energy_conservation_violation
    action: switch_to_base_calculator
```

### Python API

#### Example 1: Basic MTS with QM+ML

```python
from goal.md.mts.dual_calculator.qm_ml_learner import QMLearnerCalculator
from goal.md.mts.integrators import MTSIntegrator
from goal.md.core import MoleculeFactory

# Create molecule
atoms = MoleculeFactory.create("from_smiles", smiles="CCO")

# Setup dual calculator
base_calc = CalculatorFactory.create(
    "orca",
    orca_path="/path/to/orca"
)

ml_calc = CalculatorFactory.create(
    "goal_model",
    checkpoint="initial_model.ckpt"
)

# Create dual-calculator system
dual_calc = QMLearnerCalculator(
    base_calculator=base_calc,
    ml_calculator=ml_calc,
    force_rmse_threshold=0.01,
    evaluation_interval=100
)

atoms.calc = dual_calc

# Run MTS MD
integrator = MTSIntegrator(
    atoms=atoms,
    timestep=1.0,
    temperature_K=300,
    backend="ipi"
)

integrator.run(n_steps=10000)
```

#### Example 2: Config-based MTS Workflow

```python
from hydra import compose, initialize_config_dir
from goal.md.mts.config_parsers import build_mts_simulation

with initialize_config_dir(config_dir="configs/md/mts"):
    cfg = compose(config_name="simulations/qm_ml_adaptive")

    # Automatically builds dual calculator + integrator
    sim = build_mts_simulation(cfg)

    # Run with automatic switching
    sim.run()

    # Access metrics
    metrics = sim.get_metrics()
    print(f"ML accuracy history: {metrics['ml_rmse']}")
    print(f"Switch events: {metrics['switch_events']}")
```

#### Example 3: Monitoring & Adaptive Control

```python
from goal.md.mts.dual_calculator.calculator_switch import AdaptiveSwitcher

switcher = AdaptiveSwitcher(
    base_calc=qm_calc,
    ml_calc=ml_model,
    threshold=0.01
)

for step in range(10000):
    # Compute with both calculators
    e_qm, f_qm = base_calc.compute(atoms)
    e_ml, f_ml = ml_calc.compute(atoms)

    # Check accuracy
    force_error = np.linalg.norm(f_ml - f_qm) / len(atoms)

    # Train ML model
    ml_calc.train_step(atoms, f_qm, e_qm)

    # Decide which to use
    active_calc = switcher.decide(
        step=step,
        base_forces=f_qm,
        ml_forces=f_ml,
        ml_accuracy=force_error
    )

    # Take MD step with active calculator
    atoms.calc = active_calc
    dyn.run(1)
```

### i-pi Integration Layer

#### Adapter: i-pi ↔ goal.md

```python
# goal/md/mts/integrators/dynamics_converter.py

class IpiToGoalAdapter:
    """Convert i-pi integrators to goal.md API"""

    def __init__(self, ipi_motion, goal_atoms):
        self.motion = ipi_motion  # i-pi motion object
        self.atoms = goal_atoms   # ASE Atoms

    def step(self):
        """Execute one i-pi integration step"""
        # Update i-pi beads from ASE atoms
        self._sync_ipi_beads()

        # Run i-pi step with dual calculator
        self.motion.integrator.step()

        # Sync back to ASE
        self._sync_ase_atoms()

    def _sync_ipi_beads(self):
        """ASE → i-pi data"""
        self.motion.beads.q[:] = self.atoms.get_positions().flatten()
        self.motion.beads.p[:] = self._get_momenta().flatten()

    def _sync_ase_atoms(self):
        """i-pi → ASE data"""
        self.atoms.set_positions(
            self.motion.beads.q[:].reshape(-1, 3)
        )
        self._set_momenta(
            self.motion.beads.p[:].reshape(-1, 3)
        )
```

### Switching Strategies

#### 1. Accuracy-Based (Default)

```python
# Force & energy error below thresholds → switch to ML

class AccuracyBasedSwitcher:
    def decide(self, step, base_forces, ml_forces, base_energy, ml_energy):
        force_rmse = np.sqrt(np.mean((ml_forces - base_forces)**2))
        energy_mae = np.abs(ml_energy - base_energy)

        if (force_rmse < self.f_threshold and
            energy_mae < self.e_threshold):
            return "ML"
        else:
            return "QM"
```

#### 2. RL-Based (Advanced)

```python
# Use RL policy to decide based on accuracy history

class RLBasedSwitcher:
    def __init__(self, policy_network):
        self.policy = policy_network

    def decide(self, state_dict):
        # state: [step, ml_rmse, ml_mae, uncertainty, ...]
        state = torch.tensor(state_dict.values())
        action = self.policy(state)  # 0=QM, 1=ML
        return "ML" if action > 0.5 else "QM"
```

#### 3. Hybrid (Ensemble)

```python
# Use both calculators, blend results

class EnsembleBlender:
    def compute(self, atoms, ml_calc, qm_calc):
        e_ml, f_ml = ml_calc.compute(atoms)
        e_qm, f_qm = qm_calc.compute(atoms)

        # Blend based on uncertainty
        w_ml = 1.0 / (1.0 + self.ml_uncertainty)
        w_qm = 1.0 - w_ml

        return w_qm * e_qm + w_ml * e_ml, \
               w_qm * f_qm + w_ml * f_ml
```

### Online Training

```python
class OnlineMLTrainer:
    """Train ML model on-the-fly from QM data"""

    def __init__(self, ml_model, batch_size=32):
        self.model = ml_model
        self.buffer = ReplayBuffer(max_size=10000)

    def collect_data(self, atoms, base_forces, base_energy):
        """Add data to training buffer"""
        self.buffer.add(atoms, base_forces, base_energy)

    def training_step(self):
        """One training iteration"""
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        loss = self.model.train_on_batch(batch)

        return loss
```

### Implementation Path

**Phase 1: i-pi Integration Foundation**
- [ ] Create MTS module structure
- [ ] Implement i-pi ↔ ASE adapters
- [ ] Wrap i-pi integrators

**Phase 2: Dual-Calculator System**
- [ ] Implement base dual-calculator class
- [ ] Create QM+ML trainer
- [ ] Implement switching logic

**Phase 3: Config Support**
- [ ] Hydra config schema for MTS
- [ ] Config parser & builder
- [ ] Example configs

**Phase 4: Advanced Features**
- [ ] RL-based switching
- [ ] Ensemble blending
- [ ] Uncertainty quantification
- [ ] Checkpointing & resuming

**Phase 5: Validation & Docs**
- [ ] Unit tests
- [ ] Integration tests
- [ ] Comprehensive documentation
- [ ] Example notebooks

### Benefits

1. **Accurate Reference**: QM/expensive ML as ground truth
2. **Fast Inference**: ML model takes over once trained
3. **Adaptive Learning**: ML learns from most relevant data
4. **Energy Efficient**: Reduce expensive QM calls ~10-100x
5. **Transferable Models**: Trained models reusable in other sims
6. **Reproducibility**: Config-driven, version-controlled

### Known Challenges

1. **Data Distribution Shift**: ML trained on early sim may not generalize
   - Solution: Periodic re-evaluation, uncertainty quantification

2. **Energy Conservation**: ML may violate conservation laws
   - Solution: Physics-informed loss, energy conservation constraints

3. **Stability**: Quick switching may cause discontinuities
   - Solution: Smooth blending, fallback mechanisms

4. **Computational Cost**: Training while simulating expensive
   - Solution: Batch training, GPU acceleration, async training

5. **i-pi Compatibility**: Future i-pi updates may break adapters
   - Solution: Version pinning, adapter tests, flexibility in design

### References

- i-pi project: https://github.com/i-pi/i-pi
- FlashMD i-pi integration: https://github.com/lab-cosmo/flashmd/blob/main/src/flashmd/ipi.py
- Theoretical basis: Multiple timestep MD, online learning theory
