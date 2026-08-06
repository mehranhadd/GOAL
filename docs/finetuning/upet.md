# Fine-tuning UPET / PET

Unlike MACE and UMA, UPET is **not** fine-tuned in-process. UPET uses
metatensor `TensorMap` objects internally, and building valid,
gradient-carrying TensorMaps from `AtomicGraph` requires substantial metatensor
machinery. Since metatrain is **file-based** (no gradients flow through GOAL),
the right tool is a thin CLI that shells out to `mtt train`.

> **FUTURE WORK:** in-process UPET fine-tuning via a metatensor `TensorMap`
> adapter (build TensorMap systems from `AtomicGraph` so UPET trains inside
> `GOALModule` like MACE/UMA). This is tracked as a placeholder issue — please
> file one in the project tracker titled *"In-process UPET fine-tuning via
> metatensor TensorMap adapter"* referencing
> `src/goal/ml/cli/finetune_upet.py`.

## Install

```bash
pip install upet metatrain
```

## Workflow

1. **Export training data** to ExtXYZ (energy in info, forces in arrays):

   ```python
   from goal.ml.data.export import write_extxyz
   write_extxyz(train_dataset, "data/train.xyz")
   write_extxyz(val_dataset,   "data/val.xyz")
   ```

2. **Run the CLI** — it writes a metatrain `options.yaml` and runs `mtt train`:

   ```bash
   goal-finetune-upet \
       --model pet-omat-l \
       --train data/train.xyz \
       --val   data/val.xyz \
       --output runs/upet_finetuned/
   ```

   Use `--dry-run` to only generate `options.yaml` (inspect/edit before
   training). `--epochs` and `--lr` tune the run.

   > The generated `options.yaml` targets metatrain's PET architecture with a
   > `finetune` block pointing at the base model. Field names can differ across
   > metatrain/upet versions — inspect the file (`--dry-run`) and adjust if
   > `mtt train` complains.

3. **Use the fine-tuned model** in GOAL as an MD calculator:

   ```python
   from goal.md import CalculatorFactory
   calc = CalculatorFactory.create(
       "upet",
       checkpoint_path="runs/upet_finetuned/model.ckpt",
       variants={"energy": "finetune"},
   )
   ```

See also the [UPET calculator docs](../calculators/upet.md).
