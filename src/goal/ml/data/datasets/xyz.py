"""ExtXYZ dataset — reads ``.xyz`` files via ASE and converts to ``AtomicGraph``.

ASE is used at the dataset boundary only. Once structures are loaded and
converted to ``AtomicGraph``, ASE is never touched again inside the
training loop.
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("xyz")
class ExtXYZDataset(BaseAtomicDataset):
    """Dataset that reads extended XYZ files using ASE.

    Structures are read once at initialisation, converted to
    ``AtomicGraph``, and cached in memory. This trades RAM for
    zero per-epoch I/O overhead.

    Parameters
    ----------
    root : str or Path
        Directory containing ``.xyz`` files, or a single ``.xyz`` file.
    cutoff : float
        Cutoff radius for neighbour list construction (Angstrom).
    split : str
        Which split to load — used to find ``{split}.xyz`` inside *root*.
    energy_key, forces_key, stress_key : str or None
        Keys for energy (``atoms.info``), forces (``atoms.arrays``), and stress
        (``atoms.info``).  ``None`` uses the defaults (``energy`` / ``forces`` /
        ``stress``); each falls back to the ASE calculator when absent.
    key_mapping : dict or None
        MACE/FairChem-style mapping of canonical property → file key, e.g.
        ``{"energy": "REF_energy", "forces": "REF_forces"}``.  Overrides the
        per-key arguments.  A configured key absent from a frame raises a clear
        error naming the keys that are present.
    head : str or None
        Multihead training tag applied to every graph in this dataset.
    compute_fragment_index : bool
        Attach connected-component ``fragment_index`` labels to every
        frame (needed by the fragment-interaction module).
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
        head: str | None = None,
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
        self.head: str | None = head

        self._graphs: list[AtomicGraph] = []
        self._load()

    def _load(self) -> None:
        """Read the XYZ file and convert every frame to ``AtomicGraph``."""
        from ase.io import read

        path: Path = self.root / f"{self.split}.xyz"
        if not path.exists():
            # Fall back to root itself if it's a file
            path = self.root
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.is_dir():
            # A directory has no '{split}.xyz' inside and cannot be read as a
            # single file. Raise FileNotFoundError so the datamodule can fall
            # back to numeric splitting over the directory's contents.
            raise FileNotFoundError(
                f"No '{self.split}.xyz' found in directory {self.root}, and a "
                f"directory cannot be read as an xyz file. Point 'root' at a "
                f"single .xyz file, provide per-split files "
                f"('{self.split}.xyz'), or use train_dir/val_dir."
            )

        from goal.ml.data.keys import extract_ase_labels

        frames: typing.Any = read(str(path), index=":")
        if not isinstance(frames, list):
            frames = [frames]

        for atoms in frames:
            labels: dict[str, typing.Any] = extract_ase_labels(
                atoms, self._key_map, self._explicit_keys
            )
            graph: AtomicGraph = AtomicGraph.from_ase(
                atoms,
                cutoff=self.cutoff,
                energy=labels["energy"],
                forces=labels["forces"],
                stress=labels["stress"],
                head=self.head,
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

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> AtomicGraph:
        return self._graphs[idx]
