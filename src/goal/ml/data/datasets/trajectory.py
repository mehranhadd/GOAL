"""ASE Trajectory dataset — reads ``.traj`` files via ASE.

ASE trajectory files (``.traj``) are a common output format for molecular
dynamics simulations and geometry optimisations in ASE. This loader reads
them using ASE's ``ase.io.Trajectory`` reader and converts each frame
to an ``AtomicGraph``.
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("trajectory")
class TrajectoryDataset(BaseAtomicDataset):
    """Dataset that reads ASE trajectory (``.traj``) files.

    Structures are read once at initialisation, converted to
    ``AtomicGraph``, and cached in memory.

    Parameters
    ----------
    root : str or Path
        Directory containing ``.traj`` files, or a single file.
    cutoff : float
        Cutoff radius for neighbour list construction (Angstrom).
    split : str
        Which split to load — looks for ``{split}.traj`` inside *root*.
    energy_key, forces_key, stress_key : str or None
        Keys for total energy (``atoms.info``), forces (``atoms.arrays``), and
        stress (``atoms.info``).  ``None`` uses the defaults (``energy`` /
        ``forces`` / ``stress``).  Each falls back to the ASE calculator
        (``get_potential_energy`` / ``get_forces`` / ``get_stress``).
    key_mapping : dict or None
        MACE/FairChem-style mapping of canonical property → file key, e.g.
        ``{"energy": "REF_energy", "forces": "REF_forces"}``.  Overrides the
        per-key arguments.  A configured key that is absent from the file (and
        not provided by a calculator) raises a clear error listing the keys
        that *are* present.
    compute_fragment_index : bool
        Attach connected-component ``fragment_index`` labels to every
        frame (needed by the fragment-interaction module).  Computed once
        at load time, never in the training loop.
    fragment_covalent_cutoff : float
        Bond threshold for the fragment decomposition (Angstrom).
    """

    def __init__(
        self,
        root: str | Path,
        cutoff: float,
        split: str = "train",
        energy_key: str | None = None,
        forces_key: str | None = None,
        stress_key: str | None = None,
        key_mapping: typing.Mapping[str, str] | None = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_backend: str = "ase",
        compute_fragment_index: bool = False,
        fragment_covalent_cutoff: float = 1.8,
        fragment_scheme: str = "connected",
        fragment_charge: int = 0,
        fragment_on_failure: str = "fallback",
        fragment_smarts: typing.Sequence[str] | None = None,
        fragment_keep_groups: typing.Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            root=root,
            cutoff=cutoff,
            split=split,
            dtype=dtype,
            neighbor_list_backend=neighbor_list_backend,
            compute_fragment_index=compute_fragment_index,
            fragment_covalent_cutoff=fragment_covalent_cutoff,
            fragment_scheme=fragment_scheme,
            fragment_charge=fragment_charge,
            fragment_on_failure=fragment_on_failure,
            fragment_smarts=fragment_smarts,
            fragment_keep_groups=fragment_keep_groups,
        )
        from goal.ml.data.keys import resolve_label_keys

        self._key_map, self._explicit_keys = resolve_label_keys(
            energy_key, forces_key, stress_key, key_mapping
        )
        self._graphs: list[AtomicGraph] = []
        self._load()

    def _load(self) -> None:
        """Read the trajectory file and convert every frame."""
        from ase.io import Trajectory

        path: Path = self.root / f"{self.split}.traj"
        if not path.exists():
            path = self.root
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.is_dir():
            # A directory has no '{split}.traj' inside and is not itself a
            # readable trajectory. Raise FileNotFoundError (not a cryptic ASE
            # error) so the datamodule can fall back to numeric splitting over
            # the directory's contents.
            raise FileNotFoundError(
                f"No '{self.split}.traj' found in directory {self.root}, and a "
                f"directory cannot be read as a trajectory. Point 'root' at a "
                f"single .traj file, provide per-split files "
                f"('{self.split}.traj'), or use train_dir/val_dir."
            )

        from goal.ml.data.keys import extract_ase_labels

        traj: typing.Any = Trajectory(str(path), mode="r")

        for atoms in traj:
            labels: dict[str, typing.Any] = extract_ase_labels(
                atoms, self._key_map, self._explicit_keys
            )
            graph: AtomicGraph = AtomicGraph.from_ase(
                atoms,
                cutoff=self.cutoff,
                energy=labels["energy"],
                forces=labels["forces"],
                stress=labels["stress"],
                dtype=self.dtype,
                neighbor_list_backend=self.neighbor_list_backend,
                compute_fragment_index=self.compute_fragment_index,
                fragment_covalent_cutoff=self.fragment_covalent_cutoff,
                fragment_scheme=self.fragment_scheme,
                fragment_charge=self.fragment_charge,
                fragment_on_failure=self.fragment_on_failure,
                fragment_smarts=self.fragment_smarts,
                fragment_keep_groups=self.fragment_keep_groups,
            )
            self._graphs.append(graph)

        traj.close()

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> AtomicGraph:
        return self._graphs[idx]
