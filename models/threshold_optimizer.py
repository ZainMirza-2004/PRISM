"""Per-class decision thresholds constrained by sanity-slice FPR/FNR caps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
from sklearn.metrics import f1_score

from models.label_config import LABELS, NEUTRAL_LABEL


def predicted_class_with_thresholds(
    posteriors: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    classes_ordered: Sequence[str],
    ambiguity_margin: float | None = None,
) -> str:
    """Classes exceeding per-class thresholds; ties broken by highest probability.

    If ``ambiguity_margin`` is set (>0): when at least one *biased* class passes its threshold,
    but its mass does not exceed ``posteriors[neutral]`` by at least ``ambiguity_margin``,
    return **neutral** (abstention on overlapping neutral vs biased scores).
    """

    candidates: list[tuple[str, float]] = []
    for c in classes_ordered:
        p = float(posteriors.get(str(c), 0.0))
        if p >= float(thresholds[str(c)]):
            candidates.append((str(c), p))

    if not candidates:
        return NEUTRAL_LABEL

    biased_candidates = [(c, p) for c, p in candidates if str(c) != NEUTRAL_LABEL]
    pn = float(posteriors.get(NEUTRAL_LABEL, 0.0))

    if (
        ambiguity_margin is not None
        and ambiguity_margin > 0
        and biased_candidates
        and NEUTRAL_LABEL in classes_ordered
    ):
        bb = max(float(p) for _, p in biased_candidates)
        if bb - pn < float(ambiguity_margin):
            return NEUTRAL_LABEL

    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[0][0]


def compute_sanity_rates(
    rows: Sequence[MutableMapping[str, Any]],
    y_pred_labels: Sequence[str],
) -> tuple[float, float]:
    """FPR over neutral_policy+positive_group; FNR over explicit_or_implicit_bias."""
    fp, fp_den, fn, fn_den = 0, 0, 0, 0
    for row, lab in zip(rows, y_pred_labels):
        st = str(row.get("slice_type", "") or "").strip()
        ambiguous = str(row.get("ambiguous_stance", "0")).strip().lower() in {"1", "true", "yes"}
        if ambiguous:
            continue
        if st in {"neutral_policy", "positive_group"}:
            fp_den += 1
            if lab != NEUTRAL_LABEL:
                fp += 1
        elif st == "explicit_or_implicit_bias":
            fn_den += 1
            if lab == NEUTRAL_LABEL:
                fn += 1
    fpr = float(fp / fp_den) if fp_den else 0.0
    fnr = float(fn / fn_den) if fn_den else 0.0
    return fpr, fnr


def _penalized_f1(theta: Mapping[str, float], ambig: float, *, ys_arr, grid_classes, post_val, rows_sanity, post_sanity, fmax: float, nmax: float) -> tuple[float, float, float]:
    preds = [
        predicted_class_with_thresholds(p, theta, classes_ordered=grid_classes, ambiguity_margin=ambig)
        for p in post_val
    ]
    f1v = float(f1_score(ys_arr, preds, labels=list(grid_classes), average="macro", zero_division=0))
    preds_s = [
        predicted_class_with_thresholds(p, theta, classes_ordered=grid_classes, ambiguity_margin=ambig)
        for p in post_sanity
    ]
    fp, nr = compute_sanity_rates(rows_sanity, preds_s)
    penalty = max(0.0, fp - fmax) * 500.0 + max(0.0, nr - nmax) * 500.0
    return f1v - penalty, fp, nr


def optimize_thresholds(
    *,
    y_val_true: Sequence[str],
    calibrated_val_posteriors: Sequence[Mapping[str, float]],
    sanity_rows: Sequence[MutableMapping[str, Any]],
    calibrated_sanity_posteriors: Sequence[Mapping[str, float]],
    classes_ordered: Sequence[str] | None = None,
    fpr_max: float = 0.20,
    fnr_max: float = 0.20,
    grid_hi: float = 0.92,
    grid_lo: float = 0.08,
    step: float = 0.01,
    max_outer_rounds: int = 10,
    ambiguity_margin_max: float = 0.34,
    ambiguity_margin_step: float = 0.02,
    random_refine_iterations: int = 8000,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Tune per-class thresholds + optional neutrality gap to satisfy sanity gates.

    **Ambiguity margin** (non-negative float): prefer **neutral** when any biased label
    passes its threshold yet its calibrated mass does not exceed ``posteriors[neutral]``
    by at least this gap (fixes overlapping scorer outputs without changing the LR weights).
    Jointly optimised with thresholds under FPR/FNR caps on the sanity slice; if no
    feasible point exists under coordinate descent + margin sweep, a penalised random
    search biases toward the feasibility box while maximising validation macro-F1.
    """

    classes = tuple(classes_ordered or LABELS)
    ys = np.asarray(list(y_val_true), dtype=object)

    grid = np.round(np.arange(grid_lo, grid_hi + step * 0.25, step), 4)

    margins = np.round(np.arange(0.0, ambiguity_margin_max + ambiguity_margin_step * 0.25, ambiguity_margin_step), 4)

    def macro_f1_val(theta: Mapping[str, float], ambig: float) -> float:
        preds = [
            predicted_class_with_thresholds(p, theta, classes_ordered=classes, ambiguity_margin=ambig)
            for p in calibrated_val_posteriors
        ]
        return float(f1_score(ys, preds, labels=list(classes), average="macro", zero_division=0))

    def sanity_ok(theta: Mapping[str, float], ambig: float) -> bool:
        preds = [
            predicted_class_with_thresholds(p, theta, classes_ordered=classes, ambiguity_margin=ambig)
            for p in calibrated_sanity_posteriors
        ]
        fpr, fnr = compute_sanity_rates(sanity_rows, preds)
        return fpr <= fpr_max + 1e-9 and fnr <= fnr_max + 1e-9

    theta: dict[str, float] = {str(c): 0.50 for c in classes}

    meta: dict[str, Any] = {}
    best_amb: float = 0.0
    best_theta: dict[str, float] | None = None
    best_feasible_f1: float = -1.0

    for amb in margins:
        theta = {str(c): 0.50 for c in classes}
        for _rnd in range(max_outer_rounds):
            for cls in classes:
                best_f1_here = macro_f1_val(theta, amb)
                best_theta_here = dict(theta)
                for t in grid:
                    cand = dict(theta)
                    cand[str(cls)] = float(t)
                    if not sanity_ok(cand, amb):
                        continue
                    f1_here = macro_f1_val(cand, amb)
                    if f1_here > best_f1_here + 1e-8:
                        best_f1_here = f1_here
                        best_theta_here = cand
                theta = best_theta_here

        if sanity_ok(theta, amb):
            mf1 = macro_f1_val(theta, amb)
            if mf1 > best_feasible_f1 + 1e-12:
                best_feasible_f1 = mf1
                best_theta = dict(theta)
                best_amb = float(amb)

    if best_theta is not None and sanity_ok(best_theta, best_amb):
        meta["feasible"] = True
        meta["ambiguity_margin"] = best_amb
        meta["final_macro_f1_val"] = macro_f1_val(best_theta, best_amb)
        preds_sanity = [
            predicted_class_with_thresholds(p, best_theta, classes_ordered=classes, ambiguity_margin=best_amb)
            for p in calibrated_sanity_posteriors
        ]
        meta["sanity_fpr"], meta["sanity_fnr"] = compute_sanity_rates(sanity_rows, preds_sanity)
        meta["note"] = ""
        return best_theta, meta

    meta["feasible"] = False
    meta["note"] = "No greedy+margin feasibility; using penalised quasi-random refinement."
    meta["relaxed_fallback"] = True

    rng = np.random.default_rng(42)
    best_score = -1e100
    out_theta = {str(c): 0.5 for c in classes}
    out_amb = 0.0

    for _ in range(random_refine_iterations):
        amb = float(rng.uniform(0.0, min(ambiguity_margin_max + 0.001, 0.45)))
        cand_theta = {str(c): float(rng.choice(grid)) for c in classes}
        scr, _fp, _nr = _penalized_f1(
            cand_theta,
            amb,
            ys_arr=ys,
            grid_classes=classes,
            post_val=calibrated_val_posteriors,
            rows_sanity=sanity_rows,
            post_sanity=calibrated_sanity_posteriors,
            fmax=fpr_max,
            nmax=fnr_max,
        )
        if scr > best_score + 1e-14:
            best_score = scr
            out_theta = cand_theta
            out_amb = amb

    preds_sf = [
        predicted_class_with_thresholds(p, out_theta, classes_ordered=classes, ambiguity_margin=out_amb)
        for p in calibrated_sanity_posteriors
    ]
    meta["ambiguity_margin"] = float(out_amb)
    meta["sanity_fpr"], meta["sanity_fnr"] = compute_sanity_rates(sanity_rows, preds_sf)
    meta["final_macro_f1_val"] = macro_f1_val(out_theta, out_amb)
    meta["penalised_objective"] = float(best_score)
    if sanity_ok(out_theta, out_amb):
        meta["feasible"] = True
        meta.pop("relaxed_fallback", None)
        meta["note"] = "Greedy+margins sweep infeasible; quasi-random search found thresholds+ambiguity_gap meeting sanity caps."
    return out_theta, meta


def save_thresholds_json(path: str, thresholds: Mapping[str, float], meta: Mapping[str, Any] | None = None) -> None:
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    meta_serial = dict(meta or {})
    amb = meta_serial.get("ambiguity_margin")

    for k, v in list(meta_serial.items()):
        if hasattr(v, "item"):
            try:
                meta_serial[k] = v.item()
            except Exception:
                meta_serial[k] = str(v)
        elif isinstance(v, (np.integer, np.floating)):
            meta_serial[k] = float(v)
        elif isinstance(v, np.ndarray):
            meta_serial[k] = v.tolist()

    payload: dict[str, Any] = {"thresholds": dict(thresholds), "meta": meta_serial}
    am = amb if amb is not None else meta_serial.get("ambiguity_margin")
    if am is not None:
        payload["ambiguity_margin"] = float(am)

    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_threshold_config(path: str) -> tuple[dict[str, float], float | None]:
    """Load per-class thresholds and optional neutrality ambiguity gap (``ambiguity_margin``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    th_raw = data.get("thresholds", data)
    if not isinstance(th_raw, Mapping):
        th_raw = {}
    th_out: dict[str, float] = {}
    known = tuple(LABELS)
    for k, v in th_raw.items():
        sk = str(k)
        if sk in known:
            th_out[sk] = float(v)

    amb: float | None = None
    if isinstance(data.get("ambiguity_margin"), (int, float)):
        amb = float(data["ambiguity_margin"])
    elif isinstance(data.get("meta"), Mapping):
        mv = data["meta"].get("ambiguity_margin")
        if isinstance(mv, (int, float)):
            amb = float(mv)

    return th_out, amb


def load_thresholds_json(path: str) -> dict[str, float]:
    """Class-only thresholds dict (backward compatible); use :func:`load_threshold_config` for margin."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    th_raw = data.get("thresholds", data)
    if isinstance(th_raw, Mapping):
        return {str(k): float(v) for k, v in th_raw.items() if str(k) in LABELS}
    return {}
