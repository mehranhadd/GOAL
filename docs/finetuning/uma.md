# Fine-tuning UMA / FairChem

`uma_finetune` mirrors `mace_finetune`: wrap a pre-trained UMA model as a
trainable monolithic backbone and fine-tune it through the standard GOAL loop.

Source:
[`goal.ml.nn.models.foundation.uma.UMAFinetune`](../../src/goal/ml/nn/models/foundation/uma.py).

## Status: raw-module load is not publicly documented

fairchem-core's public API returns an **inference predictor / ASE calculator**,
not a bare trainable `nn.Module`. GOAL's wrapper attempts the most likely
documented path (`pretrained_mlip.get_predict_unit(...).model`) and, if it is
not available, raises `UnsupportedOperationError` with a link to the fairchem
docs (<https://fair-chem.github.io/>).

The rest of the wrapper — input/output adapters, `head_only`/`full`/`lora`
strategies, E0 re-estimation — is complete, so `uma_finetune` starts working as
soon as a raw-module load path is available in your fairchem version. If you
know the correct entry point for your version, adjust `UMAFinetune._load_model`
(and confirm the batch field names in `_adapt_input`).

## Dataset-type embedding: `omol` vs `omat`

UMA conditions on a **dataset-type** embedding. Choose it via the `head` key:

```yaml
model:
  backbone:
    name: uma_finetune
    checkpoint: "uma-s-1"
    head: omol      # organic molecules
    # head: omat    # inorganic materials
    strategy: head_only
    dtype: float64
    reestimate_e0s: true
  head: null
```

`_adapt_input` attaches this as a per-graph dataset field; **confirm the exact
field name against your installed fairchem version** (it is marked in the
source).

## Run

```bash
goal-train --config-name finetune/uma data.root=data/finetune
```

Everything else (checkpointing, curriculum, DDP, HPO) is identical to
[MACE fine-tuning](mace.md).
