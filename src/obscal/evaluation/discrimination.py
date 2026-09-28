"""Discrimination, direction tests and rank agreement.

Ref: Sec. External validation and the artefact-confound control; Table 4.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def roc_auc(scores: NDArray[np.float64], labels: NDArray[np.bool_]) -> float:
    """Area under the ROC curve from the rank statistic, with mid-ranks for ties."""
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    if values.shape != positive.shape:
        raise ValueError("scores and labels must align")
    n_positive = int(np.sum(positive))
    n_negative = int(values.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("an ROC curve needs both classes present")
    ranks = stats.rankdata(values, method="average")
    rank_sum = float(np.sum(ranks[positive]))
    statistic = rank_sum - n_positive * (n_positive + 1) / 2.0
    return statistic / (n_positive * n_negative)


def auc_with_ci(
    scores: NDArray[np.float64], labels: NDArray[np.bool_], level: float = 0.95
) -> dict[str, float]:
    """Hanley-McNeil interval on the rank-based area."""
    area = roc_auc(scores, labels)
    positive = np.asarray(labels, dtype=bool)
    n_positive = int(np.sum(positive))
    n_negative = int(positive.size - n_positive)
    q1 = area / (2.0 - area)
    q2 = 2.0 * area**2 / (1.0 + area)
    variance = (
        area * (1.0 - area) + (n_positive - 1) * (q1 - area**2) + (n_negative - 1) * (q2 - area**2)
    ) / (n_positive * n_negative)
    spread = float(np.sqrt(max(variance, 0.0)))
    critical = float(stats.norm.ppf(0.5 + level / 2.0))
    return {
        "auc": area,
        "standard_error": spread,
        "ci_low": max(area - critical * spread, 0.0),
        "ci_high": min(area + critical * spread, 1.0),
        "n_positive": float(n_positive),
        "n_negative": float(n_negative),
    }


def binomial_direction_test(
    recovered: int, total: int, probability: float = 0.5
) -> dict[str, float | int | bool]:
    """Two-sided exact test of a recovery count against a coin flip.

    Ref: Sec. External validation. Recovering 71 of 123 differentially expressed
    genes in the published direction is tested against chance rather than assumed.
    """
    if total <= 0:
        raise ValueError("the binomial test needs a positive denominator")
    if not 0 <= recovered <= total:
        raise ValueError("the recovered count must lie inside the denominator")
    outcome = stats.binomtest(recovered, total, probability, alternative="two-sided")
    return {
        "recovered": int(recovered),
        "total": int(total),
        "share": recovered / total,
        "expected": probability * total,
        "p_value": float(outcome.pvalue),
        "separated_from_chance": bool(outcome.pvalue < 0.05),
    }


def rank_agreement_probability(n_items: int) -> float:
    """Chance that a ranking of n items matches the reference exactly, one in n factorial."""
    if n_items < 1:
        raise ValueError("a ranking needs at least one item")
    return 1.0 / math.factorial(n_items)


def rank_agreement(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    """Whether a second ranking of the same items matches the reference ordering."""
    if set(reference) != set(candidate):
        raise ValueError("both rankings must cover the same items")
    items = list(reference)
    left = sorted(items, key=lambda item: reference[item])
    right = sorted(items, key=lambda item: candidate[item])
    concordant = left == right
    return {
        "n_items": len(items),
        "exact_match": concordant,
        "chance_probability": rank_agreement_probability(len(items)),
        "inverse_chance": float(math.factorial(len(items))),
        "reference_order": left,
        "candidate_order": right,
    }


def programme_score_ratio(
    treated: NDArray[np.float64], control: NDArray[np.float64], level: float = 0.95
) -> dict[str, float]:
    """Ratio of a module score in one arm to the control arm, with a Welch interval."""
    a = np.asarray(treated, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("a module-score ratio needs two readings per arm")
    ratio = float(np.mean(a)) / float(np.mean(b))
    log_ratio = float(np.log(ratio))
    relative_error = float(
        np.sqrt(
            (np.std(a, ddof=1) / np.sqrt(a.size) / np.mean(a)) ** 2
            + (np.std(b, ddof=1) / np.sqrt(b.size) / np.mean(b)) ** 2
        )
    )
    critical = float(stats.norm.ppf(0.5 + level / 2.0))
    return {
        "ratio": ratio,
        "ci_low": float(np.exp(log_ratio - critical * relative_error)),
        "ci_high": float(np.exp(log_ratio + critical * relative_error)),
        "n_treated": float(a.size),
        "n_control": float(b.size),
    }


def score_ratio_test(
    treated: NDArray[np.float64], control: NDArray[np.float64]
) -> dict[str, float]:
    """Welch test that accompanies a module-score ratio."""
    result = stats.ttest_ind(treated, control, equal_var=False)
    return {"t": float(result.statistic), "p_value": float(result.pvalue)}


def source_effect_auc(values: Sequence[float], labels: Sequence[bool]) -> dict[str, float]:
    """Discrimination of a binary label by a continuous readout."""
    return auc_with_ci(np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=bool))


def direction_recovery(
    log_fold_changes: NDArray[np.float64], published_directions: NDArray[np.float64]
) -> dict[str, float | int | bool]:
    """Share of a published signature recovered in the published direction."""
    observed = np.asarray(log_fold_changes, dtype=np.float64)
    published = np.asarray(published_directions, dtype=np.float64)
    if observed.shape != published.shape:
        raise ValueError("the observed and published signatures must align")
    usable = (observed != 0.0) & (published != 0.0)
    agreed = int(np.sum(np.sign(observed[usable]) == np.sign(published[usable])))
    total = int(np.sum(usable))
    result = binomial_direction_test(agreed, total)
    return {**result, "n_compared": total}


def phenotype_concordance(within_batch_auc: float, against_series_auc: float) -> dict[str, float]:
    """Phenotype concordance in house against the external series."""
    return {
        "in_house": float(within_batch_auc),
        "against_series": float(against_series_auc),
        "drop": float(within_batch_auc) - float(against_series_auc),
        "retained_fraction": (
            float(against_series_auc) / float(within_batch_auc)
            if within_batch_auc
            else float("nan")
        ),
    }


def ranking_matches_hydroxyproline(
    arm_scores: dict[str, float], arm_hydroxyproline: dict[str, float]
) -> dict[str, float | int | bool]:
    """Whether a module's arm ranking reproduces the primary matrix readout's."""
    return rank_agreement(arm_hydroxyproline, arm_scores)


def schedule_only_information(
    label_auc_before: float, label_auc_after: float
) -> dict[str, float | bool]:
    """How much label information the schedule alone carries before and after deconfounding.

    Ref: SI Table S13. A cell-free channel predicting nothing would be a rigged
    control rather than a clean one, so the before value is what makes the after
    value worth reading.
    """
    return {
        "label_auc_before_deconfounding": float(label_auc_before),
        "label_auc_after_deconfounding": float(label_auc_after),
        "collapse": float(label_auc_before) - float(label_auc_after),
        "residual_is_near_chance": bool(abs(label_auc_after - 0.5) < 0.1),
    }
