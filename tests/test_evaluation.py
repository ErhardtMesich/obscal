"""Contrasts, multiplicity, discrimination and the plain-text renderer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from obscal.evaluation import (
    arm_against_untreated,
    auc_with_ci,
    batch_adjusted_contrast,
    binomial_direction_test,
    bonferroni,
    direction_recovery,
    dispersion,
    family_from_names,
    holm_step_down,
    mann_whitney,
    phenotype_concordance,
    pooling_is_licensed,
    programme_score_ratio,
    rank_agreement,
    rank_agreement_probability,
    ranking_matches_hydroxyproline,
    relative_reduction,
    render_key_values,
    render_matrix,
    render_table,
    roc_auc,
    schedule_only_information,
    source_effect_auc,
    unadjusted_clearance,
    welch_t_test,
)


def test_welch_matches_its_hand_computed_form() -> None:
    rng = np.random.default_rng(2)
    left = rng.normal(2.0, 0.6, size=40)
    right = rng.normal(1.4, 0.9, size=52)
    contrast = welch_t_test(left, right)
    variance_left = float(np.var(left, ddof=1)) / 40
    variance_right = float(np.var(right, ddof=1)) / 52
    assert contrast.statistic == pytest.approx(
        (float(np.mean(left)) - float(np.mean(right))) / math.sqrt(variance_left + variance_right)
    )
    assert contrast.degrees_of_freedom > 0
    assert contrast.ci_low < contrast.estimate < contrast.ci_high
    with pytest.raises(ValueError):
        welch_t_test(np.array([1.0]), np.array([1.0, 2.0]))


def test_batch_adjusted_contrast_recovers_a_planted_effect() -> None:
    rng = np.random.default_rng(8)
    arms = ["treated"] * 12 + ["control"] * 12 + ["treated"] * 12 + ["control"] * 12
    batches = ["B1"] * 24 + ["B2"] * 24
    values = np.asarray([1.6] * 12 + [2.4] * 12 + [1.7] * 12 + [2.5] * 12) + rng.normal(
        scale=0.05, size=48
    )
    values[24:] -= 0.30
    adjusted = batch_adjusted_contrast(values, arms, batches, treated="treated", control="control")
    assert adjusted.estimate == pytest.approx(-0.80, abs=0.05)
    assert adjusted.significant


def test_batch_adjusted_contrast_rejects_a_missing_arm() -> None:
    with pytest.raises(ValueError):
        batch_adjusted_contrast(np.arange(4.0), ["a"] * 4, ["B1"] * 4, treated="a", control="b")


def test_mann_whitney_and_dispersion() -> None:
    contrast = mann_whitney(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert 0.0 <= contrast.p_value <= 1.0
    summary = dispersion(np.array([1.0, 2.0, 3.0]))
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["n"] == 3.0


def test_relative_reduction_and_pooling_licence() -> None:
    report = relative_reduction(2.26, 3.42)
    assert report["relative_reduction_pct"] == pytest.approx(100 * (3.42 - 2.26) / 3.42)
    assert arm_against_untreated(np.array([2.26]), np.array([3.42]), "S4")["arm"] == "S4"
    licensed = pooling_is_licensed(0.02, 0.71)
    assert licensed["exchangeable"]
    assert not pooling_is_licensed(0.5, 0.01)["exchangeable"]


def test_holm_matches_a_brute_force_step_down() -> None:
    p_values = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.6])
    result = holm_step_down(p_values)
    order = np.argsort(p_values, kind="stable")
    running = 0.0
    expected = np.empty_like(p_values)
    for rank, index in enumerate(order):
        running = max(running, p_values[index] * (p_values.size - rank))
        expected[index] = min(running, 1.0)
    assert np.allclose(result.adjusted, expected)
    assert result.n_rejected == 2
    assert bonferroni(p_values)["family_threshold"] == pytest.approx(0.05 / 6)
    assert unadjusted_clearance(p_values)["n_clearing_zero"] == 4


def test_holm_names_its_family_and_its_ties() -> None:
    named = family_from_names(["a", "b", "c"], np.array([0.001, 0.5, 0.9]))
    assert named["retained"] == ["a"]
    assert named["dropped"] == ["b", "c"]
    tied = holm_step_down(np.full(4, 0.2))
    assert tied.n_rejected == 0
    assert any(value == pytest.approx(0.8) for value in tied.adjusted)


def test_roc_auc_matches_a_pair_count() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.35, 0.2])
    labels = np.array([True, True, True, False, False, False])
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    concordant = sum(
        1
        for index in range(labels.size)
        if labels[index]
        for other in range(labels.size)
        if not labels[other] and scores[index] > scores[other]
    )
    assert roc_auc(scores, labels) == pytest.approx(concordant / (positives * negatives))
    assert auc_with_ci(scores, labels)["ci_low"] <= auc_with_ci(scores, labels)["auc"]
    with pytest.raises(ValueError):
        roc_auc(scores, np.zeros(6, dtype=bool))


def test_binomial_direction_test_matches_the_exact_sum() -> None:
    recovered, total = 74, 123
    brute = min(
        1.0,
        2.0 * sum(math.comb(total, index) for index in range(recovered, total + 1)) / 2**total,
    )
    assert float(binomial_direction_test(recovered, total)["p_value"]) == pytest.approx(brute)
    with pytest.raises(ValueError):
        binomial_direction_test(5, 0)


def test_rank_agreement_and_its_chance_level() -> None:
    assert rank_agreement_probability(4) == pytest.approx(1 / 24)
    ranking = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    agreement = rank_agreement(ranking, ranking)
    assert agreement["exact_match"]
    assert agreement["n_items"] == 4
    assert agreement["inverse_chance"] == 24.0
    with pytest.raises(ValueError):
        rank_agreement({"a": 1.0}, {"b": 1.0})


def test_programme_score_ratio_and_direction_recovery() -> None:
    rng = np.random.default_rng(4)
    treated = rng.normal(1.62, 0.2, size=20) + 1.0
    control = rng.normal(1.0, 0.2, size=20) + 1.0
    ratio = programme_score_ratio(treated, control)
    assert ratio["ci_low"] < ratio["ratio"] < ratio["ci_high"]
    recovery = direction_recovery(np.array([1.0, -1.0, -0.5, 0.2]), np.array([1.0, 1.0, -1.0, 0.0]))
    assert recovery["recovered"] == 2
    assert recovery["n_compared"] == 3


def test_schedule_only_information_and_source_effect() -> None:
    report = schedule_only_information(0.71, 0.55)
    assert report["collapse"] == pytest.approx(0.16)
    assert report["residual_is_near_chance"]
    auc = source_effect_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
    assert auc["auc"] == pytest.approx(1.0)


def test_concordance_and_arm_ranking() -> None:
    concordance = phenotype_concordance(0.81, 0.73)
    assert concordance["drop"] == pytest.approx(0.08)
    ranking = ranking_matches_hydroxyproline(
        {"S2": 3.42, "S4": 2.26, "S5": 2.44, "S1": 1.58},
        {"S2": 0.9, "S4": 0.4, "S5": 0.5, "S1": 0.1},
    )
    assert ranking["exact_match"]


def test_renderer_takes_its_widths_from_the_data() -> None:
    rows = [{"a": 1.0, "b": "a deliberately long cell value"}, {"a": 1234.0, "b": "x"}]
    rendered = render_table(rows, ["a", "b"])
    assert "a deliberately long cell value" in rendered
    assert "1234" in rendered
    assert "b" in rendered
    matrix = render_matrix([[1.0, 2.0]], ["row"], ["c1", "c2"])
    assert "c1" in matrix
    assert "value" in render_key_values({"value": 1.0})
