# Fine-tuning foundation models in GOAL

GOAL fine-tunes pre-trained interatomic potentials **in-process**: a
foundation model is wrapped as a normal GOAL *backbone* and trained through the
existing `GOALModule` + Lightning trainer. There is **no separate training
loop** — the same loss, stage curriculum, optimiser, `GOALCheckpointManager`,
multi-GPU (DDP/FSDP) and HPO all apply unchanged.

| Model | How | Backbone / entry point |
|-------|-----|------------------------|
| MACE | in-process, trainable | `mace_finetune` → `goal-train --config-name finetune/mace` |
| UMA / FairChem | in-process (when a raw-module load path exists) | `uma_finetune` → `goal-train --config-name finetune/uma` |
| UPET / PET | subprocess (metatrain) | `goal-finetune-upet` CLI |

MACE and UMA are monolithic backbones (`nn.Module`) that return
`{energy, forces, num_atoms}` directly, so their configs use `head: null`.
UPET uses metatensor `TensorMap`s internally; an in-process wrapper is out of
scope, so it is fine-tuned by shelling out to `mtt train`.

## The three fine-tuning strategies

Set via `model.backbone.strategy`:

| strategy | what trains | when to use | suggested LR |
|----------|-------------|-------------|--------------|
| `head_only` | only the final readout layers (everything else frozen) | small datasets (< ~5000 structures); fastest, most stable | ~1e-4 |
| `full` | all parameters | larger datasets; risk of catastrophic forgetting on small ones | ~1e-5 (much smaller!) |
| `lora` | frozen base + LoRA adapters injected into every `nn.Linear` (needs `peft`) | parameter-efficient adaptation; keeps the base intact | ~1e-4 |

`head_only` and `lora` freeze parameters with `requires_grad=False`; GOAL's
optimiser only builds parameter groups from trainable params, so frozen weights
cost no optimiser state.

## Re-estimate E0s — do not skip this

The **single most common cause of fine-tuning failure** is a mismatch between
the foundation model's per-element reference energies (E0s), which were fit to
*its* training distribution (e.g. MPTrj), and your dataset. Left uncorrected,
the interaction network must absorb an O(eV) per-atom offset.

GOAL fixes this automatically: with `reestimate_e0s: true` (default), the
`FoundationE0Callback` runs a ridge/least-squares regression
(`E_total = Σ_Z n_Z · e_Z`) on your **training set** and writes the result into
the model's atomic-energy buffer **before** training starts.

## Which foundation model to start from?

- **Organic molecules / drug-like chemistry** → MACE-MP or MACE-OFF; UMA with
  `head: omol`; `pet-spice`.
- **Inorganic materials / periodic solids** → MACE-MP; UMA with `head: omat`;
  `pet-omat`, `pet-mad`.
- Match the model's **cutoff** in `data.cutoff` (e.g. mace-mp-0 medium/large use
  6.0 Å).

## dtype

Foundation MACE models are typically `float64`; the wrappers cast inputs to the
model dtype in `_adapt_input` and cast predictions back to the label dtype in
`_adapt_output`. Use `precision: "64"` in the trainer to match.

See [mace.md](mace.md), [uma.md](uma.md), [upet.md](upet.md).
