"""MLIP diagnostic metrics computed at every train/val step.

Standard MLFF-paper metrics (MAE/RMSE per atom for energy, MAE/RMSE
and cosine-similarity / magnitude split for forces, Newton-violation
for translation invariance).  Computed alongside the loss so the
progress bar and loggers can show *physically meaningful* numbers
even when the loss values themselves are weighted sums of several
components.

The entry point is :func:`mlip_metrics(predictions, batch)` which
returns a flat ``{name: scalar tensor}`` dict.  The training loop
prefixes the keys with ``"train/"`` or ``"val/"`` and feeds them to
Lightning's ``log_dict``.
"""

from __future__ import annotations

import typing

import torch

# Names elevated to the Lightning progress bar.  The training loop
# reads this so it can route only the headline metrics through
# ``prog_bar=True``.
PROG_BAR_METRICS: frozenset[str] = frozenset(
    {
        "energy_mae_per_atom",
        "forces_cosine_similarity",
        "newton_violation",
    }
)


def _scatter_sum_batch(
    values: torch.Tensor,  # (N, ...)
    batch_index: torch.Tensor,  # (N,)
) -> torch.Tensor:
    """``scatter_sum`` per graph index — local helper to avoid a
    top-level import of ``torch_geometric``'s scatter for a single
    operation."""
    from torch_geometric.utils import scatter

    return scatter(values, batch_index, dim=0, reduce="sum")


def mlip_metrics(
    predictions: dict[str, torch.Tensor],
    batch: typing.Any,
) -> dict[str, torch.Tensor]:
    """Compute the standard MLIP diagnostic metrics for one batch.

    Parameters
    ----------
    predictions : dict
        Model output.  Must contain ``"energy"`` shape ``(G,)`` and
        ``"num_atoms"`` shape ``(G,)``.  ``"forces"`` shape
        ``(N, 3)`` is optional — when absent the force-side metrics
        are skipped.
    batch : AtomicGraph / pyg.Batch
        Reference labels.  Reads ``batch.energy`` (per graph) and
        ``batch.forces`` (per atom) when present.  ``batch.batch``
        is the per-atom graph index (PyG convention).

    Returns
    -------
    dict
        ``{metric_name: 0-d Tensor}``.  Metrics that can't be
        computed (e.g. forces requested but no labels available) are
        simply omitted from the dict.
    """
    out: dict[str, torch.Tensor] = {}

    # ---------- Energy: per-atom MAE / RMSE ----------
    pred_e: torch.Tensor | None = predictions.get("energy", None)
    num_atoms: torch.Tensor | None = predictions.get("num_atoms", None)
    target_e: torch.Tensor | None = getattr(batch, "energy", None)
    if pred_e is not None and target_e is not None and num_atoms is not None:
        # Flatten labels in case ``Batch`` returns ``(G, 1)``.
        t_e: torch.Tensor = target_e.to(pred_e.dtype).flatten()
        n_at: torch.Tensor = num_atoms.to(pred_e.dtype).clamp(min=1.0)
        per_atom_err: torch.Tensor = (pred_e - t_e) / n_at  # (G,)
        out["energy_mae_per_atom"] = per_atom_err.abs().mean()
        out["energy_rmse_per_atom"] = per_atom_err.pow(2).mean().sqrt()

    # ---------- Forces: MAE, RMSE, cos-sim, magnitude MAE ----------
    pred_f: torch.Tensor | None = predictions.get("forces", None)
    target_f: torch.Tensor | None = getattr(batch, "forces", None)
    if pred_f is not None and target_f is not None and pred_f.shape == target_f.shape:
        t_f: torch.Tensor = target_f.to(pred_f.dtype)
        err: torch.Tensor = pred_f - t_f  # (N, 3)
        out["forces_mae"] = err.abs().mean()
        out["forces_rmse"] = err.pow(2).mean().sqrt()

        # Per-atom cosine similarity in [-1, 1].  Uses
        # ``F.cosine_similarity`` for its well-behaved eps semantics
        # (denominator becomes ``max(‖p‖·‖t‖, eps)`` rather than
        # ``‖p‖·‖t‖ + eps``, so a near-zero numerator divided by a
        # near-zero magnitude product still produces a number close
        # to the true direction agreement instead of being squashed
        # to ~0).  Atoms whose reference force is *exactly* zero
        # (symmetric sites, or pure-energy datasets that fill
        # ``batch.forces`` with zeros) are dropped from the average:
        # they carry no directional information so including them
        # would silently bias the metric toward 0.  If every atom in
        # the batch has zero reference force the metric is omitted
        # entirely.
        norm_p: torch.Tensor = pred_f.norm(dim=-1)  # (N,)
        norm_t: torch.Tensor = t_f.norm(dim=-1)  # (N,)
        cos_sim: torch.Tensor = torch.nn.functional.cosine_similarity(
            pred_f, t_f, dim=-1, eps=1e-8
        )  # (N,) in [-1, 1]
        mask: torch.Tensor = norm_t > 0.0
        if bool(mask.any()):
            out["forces_cosine_similarity"] = cos_sim[mask].mean()

        out["forces_magnitude_mae"] = (norm_p - norm_t).abs().mean()

    # ---------- Newton violation (translational-invariance check) ----------
    #
    # ``v_m = ||Σ_i F_pred_i||`` for each molecule m, averaged over
    # the batch.  ~0 for autograd-derived forces because the energy
    # is translation invariant by construction; meaningfully > 0 for
    # direct heads, in which case it's a real diagnostic of how
    # badly translation invariance is broken.  Logged regardless of
    # whether a newton loss is configured — it's a diagnostic, not
    # a training signal.
    batch_idx: typing.Any = getattr(batch, "batch", None)
    if pred_f is not None and batch_idx is not None:
        net_force_per_graph: torch.Tensor = _scatter_sum_batch(pred_f, batch_idx)  # (G, 3)
        out["newton_violation"] = net_force_per_graph.norm(dim=-1).mean()

    return out
