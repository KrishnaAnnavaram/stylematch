"""A blinded, randomized user study: the plan, and the analysis of the ratings.

Protocol (see the README, section 8):

1. Each participant sees each study query once, in a random order.
2. For each query the participant sees two lists, "List 1" and "List 2".
   A coin flip decides which system makes which list. The sheet does not name
   the systems. The key file holds the assignment.
3. The participant rates each list from 1 to 10.
4. The analysis unblinds the ratings with the key, takes the mean difference of
   each participant (so each person counts once), and tests it with the
   Wilcoxon signed-rank test. It also gives the paired t test, Cohen's dz, the
   rank-biserial correlation and a bootstrap interval. The verdict text comes
   from these numbers. It is never fixed in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def make_plan(queries: list[dict], participants: int, systems: tuple[str, str], seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sheet, key). The sheet is blind. The key maps each list to its system."""
    if participants < 2:
        raise ValueError("a study needs at least 2 participants")
    rng = np.random.default_rng(seed)
    sheet, key = [], []
    for p in range(1, participants + 1):
        order = rng.permutation(len(queries))
        for position, qi in enumerate(order, start=1):
            q = queries[qi]
            flip = bool(rng.integers(0, 2))
            first, second = (systems[1], systems[0]) if flip else systems
            pid = f"P{p:03d}"
            sheet.append({"participant": pid, "position": position, "query_id": q["query_id"], "query": q["query"], "rating_list_1": "", "rating_list_2": ""})
            key.append({"participant": pid, "query_id": q["query_id"], "list_1_system": first, "list_2_system": second})
    return pd.DataFrame(sheet), pd.DataFrame(key)


def unblind(ratings: pd.DataFrame, key: pd.DataFrame, system_a: str, system_b: str) -> pd.DataFrame:
    merged = ratings.merge(key, on=["participant", "query_id"], how="inner", validate="one_to_one")
    if len(merged) != len(ratings):
        raise ValueError("some ratings have no row in the key")
    for col in ("rating_list_1", "rating_list_2"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["rating_list_1", "rating_list_2"])
    if ((merged[["rating_list_1", "rating_list_2"]] < 1) | (merged[["rating_list_1", "rating_list_2"]] > 10)).any().any():
        raise ValueError("ratings must be from 1 to 10")
    a_first = merged["list_1_system"] == system_a
    if not (a_first | (merged["list_2_system"] == system_a)).all():
        raise ValueError(f"system {system_a!r} is not in every key row")
    merged["rating_a"] = np.where(a_first, merged["rating_list_1"], merged["rating_list_2"])
    merged["rating_b"] = np.where(a_first, merged["rating_list_2"], merged["rating_list_1"])
    return merged[["participant", "query_id", "rating_a", "rating_b"]]


@dataclass
class StudyResult:
    system_a: str
    system_b: str
    participants: int
    ratings: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    rank_biserial: float
    t_p: float
    cohens_dz: float
    verdict: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def analyze(ratings: pd.DataFrame, key: pd.DataFrame, system_a: str, system_b: str, alpha: float = 0.05, seed: int = 42) -> StudyResult:
    paired = unblind(ratings, key, system_a, system_b)
    per_person = paired.groupby("participant")[["rating_a", "rating_b"]].mean()
    diff = (per_person["rating_a"] - per_person["rating_b"]).to_numpy()
    n = diff.size
    if n < 2:
        raise ValueError("the analysis needs at least 2 participants with ratings")

    nonzero = diff[diff != 0]
    if nonzero.size:
        w_stat, w_p = stats.wilcoxon(nonzero)
        ranks = stats.rankdata(np.abs(nonzero))
        r_plus, r_minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
        rank_biserial = float((r_plus - r_minus) / ranks.sum())
    else:
        w_p, rank_biserial = 1.0, 0.0
    sd = diff.std(ddof=1)
    t_p = float(stats.ttest_1samp(diff, 0.0).pvalue) if sd > 0 else (1.0 if diff.mean() == 0 else 0.0)
    dz = float(diff.mean() / sd) if sd > 0 else 0.0
    rng = np.random.default_rng(seed)
    boot = rng.choice(diff, size=(5000, n), replace=True).mean(axis=1)
    low, high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    if w_p < alpha and (low > 0 or high < 0):
        better = system_a if diff.mean() > 0 else system_b
        verdict = f"participants preferred {better} (Wilcoxon p = {w_p:.3g}, mean difference {diff.mean():+.2f})"
    else:
        verdict = f"no evidence of a difference at alpha = {alpha} (Wilcoxon p = {w_p:.3g})"
    return StudyResult(
        system_a, system_b, n, int(len(paired)),
        float(per_person["rating_a"].mean()), float(per_person["rating_b"].mean()),
        float(diff.mean()), low, high, float(w_p), rank_biserial, t_p, dz, verdict,
    )


def simulate_ratings(sheet: pd.DataFrame, key: pd.DataFrame, quality: dict[str, float], seed: int = 42) -> pd.DataFrame:
    """SYNTHETIC ratings for the demo: participant bias + system quality + noise, clipped to 1-10."""
    rng = np.random.default_rng(seed)
    bias = {p: rng.normal(0, 1.0) for p in sheet["participant"].unique()}
    merged = sheet.merge(key, on=["participant", "query_id"])
    out = sheet.copy()
    for col, sys_col in (("rating_list_1", "list_1_system"), ("rating_list_2", "list_2_system")):
        raw = [quality[s] + bias[p] + rng.normal(0, 1.5) for s, p in zip(merged[sys_col], merged["participant"])]
        out[col] = np.clip(np.round(raw), 1, 10).astype(int)
    return out
