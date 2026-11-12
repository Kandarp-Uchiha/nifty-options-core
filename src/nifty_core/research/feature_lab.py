"""Feature lab — fast subset evaluation for ML feature selection.

Decouples slow trade-generation from fast feature-evaluation so 100+ feature
subsets can be ranked in minutes instead of hours.

Usage flow:
    1. Once: scripts/feature_lab/build_trades_cache.py
       -> writes data/_meta/feature_lab/trades.parquet + panel.parquet
    2. Repeatable: from feature_lab import evaluate_subset, parallel_evaluate
       feed candidate subsets, get ranked metrics.

Metric: P&L-weighted score using purged-k-fold OOS predictions. Each fold
trains on past-only trades (no temporal leakage) and predicts the held-out
chronological slice. We sum realized PnL on trades the filter accepts
(p_win >= threshold), and accuracy.

Parallel: ProcessPoolExecutor across subsets. Each subset eval is ~1-3 sec
(132 trades × small XGBoost), so 8 workers tear through 100 subsets in <1 min.
"""
from __future__ import annotations

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


from nifty_core.paths import PROJECT_ROOT as REPO_ROOT  # noqa: E402
CACHE_DIR = REPO_ROOT / "data" / "_meta" / "feature_lab"
TRADES_PATH = CACHE_DIR / "trades.parquet"
PANEL_PATH = CACHE_DIR / "panel.parquet"


@dataclass
class EvalResult:
    """Result of evaluating one feature subset."""
    subset: tuple[str, ...]
    n_features: int
    cv_auc: float           # mean ROC-AUC across folds
    cv_acc: float           # mean accuracy across folds
    accepted_trades: int    # n trades where pred_win >= threshold across all folds
    accepted_pnl_inr: float # sum of pnl on accepted trades (the strategy-relevant metric)
    baseline_pnl_inr: float # sum of pnl on ALL trades (what the strategy would do without filter)
    lift_inr: float         # accepted_pnl - baseline_pnl (negative = filter hurts)
    win_rate_accepted: float
    win_rate_baseline: float

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["subset"] = list(self.subset)
        return d


def _purged_kfold_indices(n: int, k: int = 5, embargo: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-ordered K-fold with embargo gap. Trades are assumed sorted by entry_date.

    For fold i:  train = indices outside [test_start - embargo, test_end + embargo]
    Prevents adjacent-day label leak (vol-surface features change slowly day-over-day).
    """
    fold_size = n // k
    splits = []
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n
        test_idx = np.arange(test_start, test_end)
        train_mask = np.ones(n, dtype=bool)
        train_mask[max(0, test_start - embargo): min(n, test_end + embargo)] = False
        train_idx = np.where(train_mask)[0]
        if len(train_idx) >= 30 and len(test_idx) >= 5:
            splits.append((train_idx, test_idx))
    return splits


def _train_xgb(X_train: pd.DataFrame, y_train: pd.Series, *, regularized: bool = True) -> XGBClassifier:
    """Train an XGBoost classifier with regularization defaults to reduce overfit."""
    if regularized:
        model = XGBClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.10,
            subsample=0.7, colsample_bytree=0.7,
            min_child_weight=5, reg_lambda=2.0, reg_alpha=0.5,
            eval_metric="logloss", verbosity=0, random_state=42, n_jobs=1,
        )
    else:
        # Match the production v1.1 config for reproducibility runs.
        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, random_state=42, n_jobs=1,
        )
    model.fit(X_train, y_train)
    return model


def _train_xgb_regressor(X_train: pd.DataFrame, y_train: pd.Series, *, regularized: bool = True) -> XGBRegressor:
    """Train an XGBoost REGRESSOR predicting trade P&L magnitude (in rupees per 1 lot).

    Used by evaluate_subset_regression: trade is gated by predicted P&L vs cost floor,
    not by binary win/loss probability. Cost floor adapts naturally to lot size since
    the regression predicts continuous expected value.
    """
    if regularized:
        model = XGBRegressor(
            n_estimators=80, max_depth=2, learning_rate=0.10,
            subsample=0.7, colsample_bytree=0.7,
            min_child_weight=5, reg_lambda=2.0, reg_alpha=0.5,
            objective="reg:squarederror",
            verbosity=0, random_state=42, n_jobs=1,
        )
    else:
        model = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror",
            verbosity=0, random_state=42, n_jobs=1,
        )
    model.fit(X_train, y_train)
    return model


def evaluate_subset_regression(
    trades: pd.DataFrame,
    feature_cols: tuple[str, ...],
    *,
    cost_floor_inr: float = 0.0,
    k_folds: int = 5,
    regularized: bool = True,
) -> EvalResult:
    """Same as evaluate_subset but uses an XGB REGRESSOR predicting expected P&L
    per trade, gated on predicted P&L >= cost_floor_inr.

    Compared to the binary classifier, regression naturally adapts to changes in
    cost structure (e.g. different lot sizes) — just raise the cost_floor_inr
    to reflect higher fixed costs at larger sizing.
    """
    df = trades.sort_values("entry_date").reset_index(drop=True)
    feature_cols = tuple(c for c in feature_cols if c in df.columns)
    if not feature_cols:
        return EvalResult(feature_cols, 0, float("nan"), float("nan"), 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    X_all = df[list(feature_cols)].astype(float)
    fill = X_all.median(numeric_only=True)
    X_all = X_all.fillna(fill)
    y_all = df["pnl_inr"].astype(float)            # regression target = realized P&L per trade
    y_win = df["win"].astype(int)                  # for win-rate reporting
    pnl_all = y_all.values

    splits = _purged_kfold_indices(len(df), k=k_folds, embargo=3)
    pred_pnl = np.full(len(df), np.nan)
    rmses = []
    for train_idx, test_idx in splits:
        X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_tr, y_te = y_all.iloc[train_idx], y_all.iloc[test_idx]
        if len(y_tr) < 20:
            continue
        model = _train_xgb_regressor(X_tr, y_tr, regularized=regularized)
        pred = model.predict(X_te)
        pred_pnl[test_idx] = pred
        rmses.append(float(np.sqrt(np.mean((pred - y_te.values) ** 2))))

    accepted_mask = pred_pnl >= cost_floor_inr
    n_acc = int(accepted_mask.sum())
    return EvalResult(
        subset=tuple(feature_cols),
        n_features=len(feature_cols),
        cv_auc=float("nan"),                                       # not applicable for regression
        cv_acc=float(np.mean(rmses)) if rmses else float("nan"),   # report RMSE in cv_acc slot
        accepted_trades=n_acc,
        accepted_pnl_inr=float(pnl_all[accepted_mask].sum()),
        baseline_pnl_inr=float(pnl_all.sum()),
        lift_inr=float(pnl_all[accepted_mask].sum()) - float(pnl_all.sum()),
        win_rate_accepted=float(y_win.values[accepted_mask].mean()) if n_acc else 0.0,
        win_rate_baseline=float(y_win.mean()),
    )


def evaluate_subset_walk_forward(
    trades: pd.DataFrame,
    feature_cols: tuple[str, ...],
    *,
    window_size: int = 100,
    refit_freq: int = 1,
    min_train: int = 30,
    threshold: float = 0.55,
    regularized: bool = True,
) -> EvalResult:
    """Walk-forward evaluation: for each trade t, retrain on the trailing
    `window_size` trades that closed BEFORE t. Refit every `refit_freq` trades.

    Trades earlier than `min_train` (warmup) are auto-accepted (mirrors the
    production IronCondorMLFilter's "return True when no model" behavior).

    This is strictly causal (no peeking at future trade outcomes).
    """
    df = trades.sort_values("entry_date").reset_index(drop=True)
    feature_cols = tuple(c for c in feature_cols if c in df.columns)
    if not feature_cols:
        return EvalResult(feature_cols, 0, float("nan"), float("nan"), 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    X_all = df[list(feature_cols)].astype(float)
    fill = X_all.median(numeric_only=True)
    X_all = X_all.fillna(fill)
    y_all = df["win"].astype(int)
    pnl_all = df["pnl_inr"].astype(float).values
    n = len(df)

    pred_proba = np.full(n, np.nan)
    # Warmup trades: auto-accept (proba = 1.0)
    pred_proba[:min_train] = 1.0

    model = None
    last_fit_at = -1
    for i in range(min_train, n):
        need_refit = (model is None) or ((i - last_fit_at) >= refit_freq)
        if need_refit:
            train_start = max(0, i - window_size)
            X_tr = X_all.iloc[train_start:i]
            y_tr = y_all.iloc[train_start:i]
            if y_tr.nunique() < 2:
                # Can't train on single-class window — auto-accept this trade
                pred_proba[i] = 1.0
                continue
            model = _train_xgb(X_tr, y_tr, regularized=regularized)
            last_fit_at = i
        pred_proba[i] = float(model.predict_proba(X_all.iloc[[i]])[0, 1])

    accepted_mask = pred_proba >= threshold
    n_acc = int(accepted_mask.sum())
    # In-sample AUC across post-warmup predictions (not OOS — for diagnostics only)
    aucs, accs = [], []
    post = pred_proba[min_train:]
    y_post = y_all.values[min_train:]
    if np.isfinite(post).any() and len(np.unique(y_post[np.isfinite(post)])) >= 2:
        from sklearn.metrics import accuracy_score, roc_auc_score
        valid = np.isfinite(post)
        try:
            aucs.append(roc_auc_score(y_post[valid], post[valid]))
        except ValueError:
            pass
        accs.append(accuracy_score(y_post[valid], (post[valid] >= 0.5).astype(int)))

    return EvalResult(
        subset=tuple(feature_cols),
        n_features=len(feature_cols),
        cv_auc=float(np.mean(aucs)) if aucs else float("nan"),
        cv_acc=float(np.mean(accs)) if accs else float("nan"),
        accepted_trades=n_acc,
        accepted_pnl_inr=float(pnl_all[accepted_mask].sum()),
        baseline_pnl_inr=float(pnl_all.sum()),
        lift_inr=float(pnl_all[accepted_mask].sum()) - float(pnl_all.sum()),
        win_rate_accepted=float(y_all.values[accepted_mask].mean()) if n_acc else 0.0,
        win_rate_baseline=float(y_all.mean()),
    )


def evaluate_subset(
    trades: pd.DataFrame,
    feature_cols: tuple[str, ...],
    *,
    threshold: float = 0.55,
    k_folds: int = 5,
    regularized: bool = True,
) -> EvalResult:
    """Evaluate a single feature subset with purged-k-fold CV.

    Required columns in `trades`: `entry_date` (sortable), `win` (0/1), `pnl_inr`,
    plus all columns in `feature_cols`.
    """
    df = trades.sort_values("entry_date").reset_index(drop=True)
    feature_cols = tuple(feature_cols)
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return EvalResult(feature_cols, 0, float("nan"), float("nan"), 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    X_all = df[list(available)].astype(float)
    fill = X_all.median(numeric_only=True)
    X_all = X_all.fillna(fill)
    y_all = df["win"].astype(int)
    pnl_all = df["pnl_inr"].astype(float).values

    aucs, accs = [], []
    pred_proba = np.full(len(df), np.nan)
    splits = _purged_kfold_indices(len(df), k=k_folds, embargo=3)
    if not splits:
        return EvalResult(feature_cols, len(available), float("nan"), float("nan"), 0, 0.0, float(pnl_all.sum()), 0.0, 0.0, float(y_all.mean()))

    from sklearn.metrics import roc_auc_score, accuracy_score
    for train_idx, test_idx in splits:
        X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_tr, y_te = y_all.iloc[train_idx], y_all.iloc[test_idx]
        if y_tr.nunique() < 2:
            continue
        model = _train_xgb(X_tr, y_tr, regularized=regularized)
        proba = model.predict_proba(X_te)[:, 1]
        pred_proba[test_idx] = proba
        try:
            aucs.append(roc_auc_score(y_te, proba))
        except ValueError:
            pass
        accs.append(accuracy_score(y_te, (proba >= 0.5).astype(int)))

    accepted_mask = pred_proba >= threshold
    accepted_trades = int(accepted_mask.sum())
    accepted_pnl = float(pnl_all[accepted_mask].sum())
    baseline_pnl = float(pnl_all.sum())
    win_acc = float(y_all.values[accepted_mask].mean()) if accepted_trades else 0.0
    win_base = float(y_all.mean())

    return EvalResult(
        subset=tuple(available),
        n_features=len(available),
        cv_auc=float(np.mean(aucs)) if aucs else float("nan"),
        cv_acc=float(np.mean(accs)) if accs else float("nan"),
        accepted_trades=accepted_trades,
        accepted_pnl_inr=accepted_pnl,
        baseline_pnl_inr=baseline_pnl,
        lift_inr=accepted_pnl - baseline_pnl,
        win_rate_accepted=win_acc,
        win_rate_baseline=win_base,
    )


# ---- Parallel runner -----------------------------------------------------

def _evaluate_subset_worker(args: tuple) -> EvalResult:
    """ProcessPoolExecutor target. Loads cache fresh in each worker."""
    subset, threshold, k_folds, regularized = args
    trades = pd.read_parquet(TRADES_PATH)
    return evaluate_subset(trades, subset, threshold=threshold, k_folds=k_folds, regularized=regularized)


def parallel_evaluate(
    subsets: Iterable[tuple[str, ...]],
    *,
    threshold: float = 0.55,
    k_folds: int = 5,
    regularized: bool = True,
    n_workers: int | None = None,
) -> list[EvalResult]:
    """Evaluate many subsets in parallel via ProcessPoolExecutor."""
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 2)
    subsets = list(subsets)
    if not subsets:
        return []
    args_list = [(s, threshold, k_folds, regularized) for s in subsets]
    results: list[EvalResult] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_evaluate_subset_worker, a): a[0] for a in args_list}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                subset = futs[fut]
                print(f"FAIL on subset {subset}: {e}")
    return results


# ---- Search strategies ---------------------------------------------------

def greedy_forward_selection(
    trades: pd.DataFrame,
    candidate_features: list[str],
    *,
    seed_features: list[str] | None = None,
    max_features: int | None = None,
    threshold: float = 0.55,
    k_folds: int = 5,
    metric: str = "lift_inr",          # "lift_inr" | "cv_auc" | "accepted_pnl_inr"
    verbose: bool = True,
) -> tuple[list[str], list[EvalResult]]:
    """Greedy forward selection: start from `seed_features` (may be empty),
    repeatedly add the single feature that maximizes `metric`.

    Returns (best_subset, trace_of_eval_results_at_each_step).
    """
    selected = list(seed_features or [])
    remaining = [f for f in candidate_features if f not in selected]
    trace: list[EvalResult] = []

    # Score seed
    if selected:
        r0 = evaluate_subset(trades, tuple(selected), threshold=threshold, k_folds=k_folds)
        trace.append(r0)
        best_metric = getattr(r0, metric)
        if verbose:
            print(f"[seed]    n={r0.n_features:>2d}  {metric}={best_metric:+.2f}  AUC={r0.cv_auc:.3f}  acc={r0.cv_acc:.3f}  features={list(r0.subset)}")
    else:
        best_metric = float("-inf")

    cap = max_features if max_features else len(candidate_features)
    while remaining and len(selected) < cap:
        # Test adding each remaining feature
        candidates = [tuple(selected + [f]) for f in remaining]
        results = parallel_evaluate(candidates, threshold=threshold, k_folds=k_folds)
        results.sort(key=lambda r: getattr(r, metric), reverse=True)
        best = results[0]
        gain = getattr(best, metric) - best_metric
        if gain <= 0:
            if verbose:
                print(f"[stop]    best add {best.subset[-1]} gives {metric}={getattr(best, metric):+.2f} (no improvement). Stopping.")
            break
        # Adopt best
        new_feat = best.subset[-1]
        selected.append(new_feat)
        remaining.remove(new_feat)
        trace.append(best)
        best_metric = getattr(best, metric)
        if verbose:
            print(f"[step {len(selected):>2d}] add {new_feat:>30s}  {metric}={best_metric:+10.2f}  AUC={best.cv_auc:.3f}  acc={best.cv_acc:.3f}  n_acc={best.accepted_trades:>3d}")

    return selected, trace


def backward_elimination(
    trades: pd.DataFrame,
    full_features: list[str],
    *,
    threshold: float = 0.55,
    k_folds: int = 5,
    metric: str = "lift_inr",
    verbose: bool = True,
) -> tuple[list[str], list[EvalResult]]:
    """Start with full set, drop features one-at-a-time if it improves metric."""
    selected = list(full_features)
    trace: list[EvalResult] = []

    r0 = evaluate_subset(trades, tuple(selected), threshold=threshold, k_folds=k_folds)
    trace.append(r0)
    best_metric = getattr(r0, metric)
    if verbose:
        print(f"[full]    n={r0.n_features:>2d}  {metric}={best_metric:+.2f}  AUC={r0.cv_auc:.3f}")

    while len(selected) > 1:
        candidates = [tuple(s for s in selected if s != f) for f in selected]
        results = parallel_evaluate(candidates, threshold=threshold, k_folds=k_folds)
        results.sort(key=lambda r: getattr(r, metric), reverse=True)
        best = results[0]
        gain = getattr(best, metric) - best_metric
        if gain <= 0:
            if verbose:
                print(f"[stop]    dropping any feature reduces {metric}. Done at n={len(selected)}.")
            break
        dropped = next(f for f in selected if f not in best.subset)
        selected = list(best.subset)
        trace.append(best)
        best_metric = getattr(best, metric)
        if verbose:
            print(f"[drop]    -{dropped:>30s}  {metric}={best_metric:+10.2f}  n={len(selected):>2d}")

    return selected, trace


def permutation_importance(
    trades: pd.DataFrame,
    feature_set: list[str],
    *,
    threshold: float = 0.55,
    k_folds: int = 5,
    n_shuffles: int = 5,
    metric: str = "cv_auc",
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance: shuffle each feature, measure metric drop.

    Returns a DataFrame ranked by importance (higher drop = more important).
    """
    rng = np.random.default_rng(seed)
    baseline = evaluate_subset(trades, tuple(feature_set), threshold=threshold, k_folds=k_folds)
    base_score = getattr(baseline, metric)
    rows = []
    for feat in feature_set:
        drops = []
        for _ in range(n_shuffles):
            scrambled = trades.copy()
            scrambled[feat] = rng.permutation(scrambled[feat].values)
            r = evaluate_subset(scrambled, tuple(feature_set), threshold=threshold, k_folds=k_folds)
            drops.append(base_score - getattr(r, metric))
        rows.append({"feature": feat, "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_results_csv(results: list[EvalResult], path: Path) -> None:
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    df["subset"] = df["subset"].apply(lambda x: "|".join(x))
    df.sort_values("lift_inr", ascending=False, inplace=True)
    df.to_csv(path, index=False)
