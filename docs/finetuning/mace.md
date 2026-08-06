# Fine-tuning MACE

`mace_finetune` loads a pre-trained MACE model as a trainable `nn.Module` and
exposes it as a monolithic GOAL backbone returning `{energy, forces,
num_atoms}`. It rides the standard `GOALModule` + Lightning trainer, so
checkpoints, curriculum, DDP and HPO all work unchanged.

Source:
[`goal.ml.nn.models.foundation.mace.MACEFinetune`](../../src/goal/ml/nn/models/foundation/mace.py).

## Install MACE (standalone venv)

`mace-torch` pins `e3nn==0.4.4`, which conflicts with GOAL's own equivariant
blocks (`e3nn>=0.5`). **Do not** install it into the pixi env. Use a dedicated
venv (see the "MACE (mace-torch)" note in `pyproject.toml`):

```bash
uv venv --python 3.14 .venv-mace
uv pip install --python .venv-mace/bin/python torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-mace/bin/python -e .
uv pip install --python .venv-mace/bin/python "mace-torch>=0.3.16"
# for LoRA:
uv pip install --python .venv-mace/bin/python peft
```

Run fine-tuning with that interpreter.

## Step by step

1. **Prepare data** as ExtXYZ with `energy` (info) and `forces` (arrays). To
   export a GOAL dataset:

   ```python
   from goal.ml.data.export import write_extxyz
   write_extxyz(my_dataset, "data/finetune/train.xyz")
   ```

2. **Pick the config** [`configs/ml/finetune/mace.yaml`](../../configs/ml/finetune/mace.yaml)
   and set `data.root`, `data.cutoff` (**must equal the checkpoint r_max** —
   6.0 Å for mace-mp-0 medium/large), and the checkpoint/strategy.

3. **Run** with the MACE venv:

   ```bash
   .venv-mace/bin/goal-train --config-name finetune/mace \
       data.root=data/finetune model.backbone.strategy=head_only
   ```

4. **Use the result** — the checkpoint is saved by `GOALCheckpointManager` like
   any GOAL model, so it loads through the normal calculator:

   ```python
   from goal.ml.utils.calculator import GOALCalculator
   calc = GOALCalculator(checkpoint_path="logs/.../checkpoints", checkpoint="best")
   ```

   (Reconstruction re-instantiates `mace_finetune`, which reloads the base MACE
   architecture and then overlays the fine-tuned weights — the base checkpoint
   must be resolvable at load time.)

## Strategy knobs

```yaml
model:
  backbone:
    name: mace_finetune
    checkpoint: "mace-mp-0-medium"   # or a path to a .model/.pt
    strategy: head_only              # head_only | full | lora
    lora_rank: 4                     # lora only
    lora_alpha: 16.0
    lora_target_modules: null        # null = all nn.Linear
    dtype: float64
    reestimate_e0s: true
```

- `head_only` freezes everything except `readouts.*`.
- `full` trains all params — drop `training.optimizer.lr` to ~1e-5.
- `lora` freezes the base and injects LoRA (`pip install peft`).

## What GOAL handles for you

- **E0 re-estimation** (`reestimate_e0s: true`) via `FoundationE0Callback` —
  re-baselines `atomic_energies_fn.atomic_energies` to your training set.
- **dtype alignment** — inputs cast to the model dtype, outputs back to labels.
- **AtomicGraph → MACE batch** — one-hot `node_attrs` over the model z-table,
  Cartesian `shifts = unit_shifts @ cell`, matching `mace.data.AtomicData`.

## DDP with a frozen backbone

`head_only`/`lora` leave many parameters unused in the graph. Under DDP set:

```yaml
strategy:
  find_unused_parameters: true
```
