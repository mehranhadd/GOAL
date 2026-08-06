# Data loading

GOAL loads atomistic datasets through [`GOALDataModule`](../../src/goal/ml/data/datamodule.py),
a Lightning `DataModule`. The `dataset_type` config key selects the reader; the
shape of the `data:` block selects the **loading mode**. Every reader converts
frames to [`AtomicGraph`](../../src/goal/ml/data/graph.py) once at load time and
caches them in memory.

## Supported formats (`dataset_type`)

| `dataset_type` | Extensions | Reader |
|----------------|-----------|--------|
| `trajectory` | `.traj` | `ase.io.Trajectory` ([trajectory.py](../../src/goal/ml/data/datasets/trajectory.py)) |
| `xyz` | `.xyz`, `.extxyz` | `ase.io.read` ([xyz.py](../../src/goal/ml/data/datasets/xyz.py)) |
| `hdf5` | `.h5`, `.hdf5` | [hdf5.py](../../src/goal/ml/data/datasets/hdf5.py) |
| `lmdb` | `.lmdb` | [lmdb.py](../../src/goal/ml/data/datasets/lmdb.py) |
| `md17`/`qm9`/`ani1`/`spice` | download/cache | `examples.datasets` |

Energy is read from `atoms.info[energy_key]`, falling back to
`atoms.get_potential_energy()`; forces from `atoms.arrays[forces_key]`, falling
back to `atoms.get_forces()` — so ASE-calculator trajectories (energy/forces on
the calculator, not in `info`/`arrays`) load correctly.

### Custom label keys (`key_mapping`)

If your file stores labels under different names, map them onto GOAL's
canonical properties — MACE/FairChem-style — with a single `key_mapping` dict
(works for `trajectory`, `xyz`, and `lmdb`):

```yaml
data:
  dataset_type: xyz
  root: /path/to/data.xyz
  key_mapping:
    energy: REF_energy     # atoms.info["REF_energy"]
    forces: REF_forces     # atoms.arrays["REF_forces"]
    stress: virial         # atoms.info["virial"]
```

`key_mapping` overrides the per-key form (`energy_key`/`forces_key`/`stress_key`).
A configured key that is absent from a frame — and not provided by a
calculator — raises a **clear error naming the keys that are present**, instead
of silently dropping the label (which used to surface later as a cryptic
`KeyError: 'energy'` in the loss). Unconfigured properties stay optional
(absent → `None`).

## Loading modes

### Mode 1 — single file, auto-split
`root` is one file; the file is loaded once and split by `split_ratio`.

```yaml
data:
  dataset_type: trajectory
  root: ${paths.data_dir}/GMD/.../O2C2H4_PBE.traj
  cutoff: 5.0
  batch_size: 8
  split_ratio: [0.8, 0.1, 0.1]   # train/val/test; [0.9, 0.1] also allowed
  split_seed: 42
```
- **Splitting:** `random_split` with `split_seed` (reproducible). Lengths always sum to the total (no frames lost); val/test are floored and the remainder goes to **train**, so a tiny dataset keeps all its frames in train. A positive ratio that rounds to 0 frames emits a warning.
- **Caveat:** this used to silently load the *whole* file as train **and** val **and** test (data leakage) because the loader fell back to the file for each split — now fixed: file roots always go through the numeric split.

### Mode 2 — pre-split files (explicit paths)
Separate `train_paths` / `val_paths` (+ optional `test_paths`). No ratio split.

```yaml
data:
  dataset_type: xyz
  train_paths: [${paths.data_dir}/train_a.xyz, ${paths.data_dir}/train_b.xyz]
  val_paths:   [${paths.data_dir}/val.xyz]
  test_paths:  [${paths.data_dir}/test.xyz]   # optional
  merge_strategy: sequential                  # or "random"
  cutoff: 5.0
  batch_size: 8
```

### Mode 3 — list of files, auto-split
`root` is a **list** — all files are merged, then split by `split_ratio`.

```yaml
data:
  dataset_type: trajectory
  root:
    - ${paths.data_dir}/C2H6_PBE.traj
    - ${paths.data_dir}/C3H8_PBE.traj
  split_ratio: [0.8, 0.1, 0.1]
  merge_strategy: sequential
  cutoff: 5.0
```
A **directory** given as `root` behaves the same way: it is expanded to all
recognised data files inside it and numeric-split (a directory with no
`{split}.<ext>` files and no data files raises a clear error).

### Mode 4 — per-split directories
`train_dir` / `val_dir` (+ optional `test_dir`); every data file in each
directory is discovered and merged into that split.

```yaml
data:
  dataset_type: trajectory
  train_dir: ${paths.data_dir}/GMD/FragmentDuplication/Training
  val_dir:   ${paths.data_dir}/GMD/FragmentDuplication/TestID
  test_dir:  ${paths.data_dir}/GMD/FragmentDuplication/TestOOD   # optional
  cutoff: 5.0
  batch_size: 48
```
`val_dir` is required when `train_dir` is set.

### Named per-split files inside one directory
If `root` is a directory containing `train.<ext>`/`val.<ext>`(/`test.<ext>`),
those named files are used directly (no ratio split).

## DataLoader / `num_workers`
- Batching uses PyG's `DataLoader` (variable atom/edge counts handled).
- **`num_workers > 0`:** Python 3.14 defaults multiprocessing to `forkserver`
  on Linux, which makes torch DataLoader workers fail with
  `ValueError: too many fds`. `GOALDataModule` therefore sets a **`fork`**
  multiprocessing context automatically for worker loaders (safe for CPU data
  loading). Override with `data.multiprocessing_context` if needed.

```yaml
data:
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 2
  # multiprocessing_context: fork   # auto on Linux; set explicitly to override
```

## Dataset-derived statistics
Computed in the training entry point ([train.py](../../src/goal/ml/cli/train.py))
from the **train split** after `setup("fit")`:
`elements` (`compute_unique_elements`), atomic reference energies / E0s
(`compute_atomic_references`, when the backbone requests them), and
`avg_num_neighbors` (`compute_avg_num_neighbors`). Foundation-model backbones
re-estimate E0s via `FoundationE0Callback` instead.

## dtype & device
Frames load as **float64** on CPU throughout the pipeline; no premature GPU
placement (device transfer is Lightning's job). The GPU neighbour-list backend
is opt-in (`neighbor_list_backend: nvalchemiops`); the default `ase` backend is
CPU and PBC-correct.

## Known limitations
- Readers are **eager** (all frames converted + cached in RAM at load) — fine
  for GMD-scale data; very large trajectories (≫10⁴ frames) will use
  proportional memory.
- A single malformed frame aborts the load (energy/forces/stress extraction is
  tolerant, but graph construction is not).
- Very small datasets can yield empty val/test splits (a warning is emitted).
