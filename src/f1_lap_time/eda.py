"""Descriptive statistics for the F1-04 train subset."""

import numpy as np
import pandas as pd

EVENT_KEYS = ["Year", "EventName", "Session"]
MIN_GROUP_ROWS = 30
REFERENCE_COMPOUND = "MEDIUM"
IQR_MULTIPLIER = 1.5
FEATURE_COLUMNS = [
    "feature",
    *EVENT_KEYS,
    "compound",
    "reference_compound",
    "n",
    "reference_n",
    "statistic",
    "value",
]


def prepare_train(train: pd.DataFrame, target: str, placeholders: list[str]) -> pd.DataFrame:
    """Normalize analysis columns and flag target tails within each event session."""
    if train.empty:
        raise ValueError("Cannot analyze an empty train subset")

    lap_time = train[target]
    if (
        not pd.api.types.is_numeric_dtype(lap_time)
        or lap_time.isna().any()
        or not np.isfinite(lap_time).all()
        or (lap_time <= 0).any()
    ):
        raise ValueError(f"{target} must be numeric, finite and positive on train")

    tyre_life = train["TyreLife"].dropna()
    if (
        not pd.api.types.is_numeric_dtype(tyre_life)
        or not np.isfinite(tyre_life).all()
        or (tyre_life < 1).any()
    ):
        raise ValueError("Non-missing TyreLife must be numeric, finite and at least 1 on train")

    analysis = train.assign(
        Compound=train["Compound"].mask(train["Compound"].isin(placeholders)),
        TyreLife_start=train["TyreLife"] - 1.0,
    )
    grouped = analysis.groupby(EVENT_KEYS, sort=False)[target]
    q1 = grouped.transform("quantile", q=0.25)
    q3 = grouped.transform("quantile", q=0.75)
    width = IQR_MULTIPLIER * (q3 - q1)
    return analysis.assign(
        below_event_fence=lap_time < q1 - width,
        above_event_fence=lap_time > q3 + width,
    )


def target_statistics(rows: pd.DataFrame, target: str) -> dict:
    lap_time = rows[target]
    quantiles = lap_time.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "n": len(rows),
        "min_s": lap_time.min(),
        "p01_s": quantiles[0.01],
        "q1_s": quantiles[0.25],
        "median_s": quantiles[0.5],
        "mean_s": lap_time.mean(),
        "q3_s": quantiles[0.75],
        "p99_s": quantiles[0.99],
        "max_s": lap_time.max(),
        "std_s": lap_time.std(),
        "iqr_s": quantiles[0.75] - quantiles[0.25],
        "below_event_fence_n": rows["below_event_fence"].sum(),
        "above_event_fence_n": rows["above_event_fence"].sum(),
        "missing_compound_n": rows["Compound"].isna().sum(),
        "missing_tyre_life_n": rows["TyreLife_start"].isna().sum(),
    }


def summarize_target(train: pd.DataFrame, target: str) -> pd.DataFrame:
    overall = {"group": "all_train", "Year": pd.NA, **target_statistics(train, target)}
    seasons = [
        {"group": "season", "Year": year, **target_statistics(rows, target)}
        for year, rows in train.groupby("Year")
    ]
    return pd.DataFrame([overall, *seasons])


def summarize_groups(train: pd.DataFrame, target: str) -> pd.DataFrame:
    summaries = []
    for event, rows in train.groupby(EVENT_KEYS):
        context = dict(zip(EVENT_KEYS, event, strict=True))
        summaries.append(
            {"group": "event", **context, "Compound": pd.NA, **target_statistics(rows, target)}
        )
        for compound, compound_rows in rows.groupby("Compound", dropna=False):
            summaries.append(
                {
                    "group": "event_compound",
                    **context,
                    "Compound": compound,
                    **target_statistics(compound_rows, target),
                }
            )
    return pd.DataFrame(summaries)


def numeric_comparison(rows: pd.DataFrame, feature: str, target: str) -> dict | None:
    paired = rows[[feature, target]].dropna()
    if len(paired) < MIN_GROUP_ROWS or (paired.nunique() < 2).any():
        return None
    return {
        "feature": feature,
        "n": len(paired),
        "statistic": "pearson_r",
        "value": paired[feature].corr(paired[target]),
    }


def compare_features(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """Measure within-event contrasts and linear associations using eligible rows."""
    comparisons = []
    for event, event_rows in train.groupby(EVENT_KEYS):
        context = dict(zip(EVENT_KEYS, event, strict=True))
        lap_number = numeric_comparison(event_rows, "LapNumber", target)
        if lap_number is not None:
            comparisons.append({**context, **lap_number})

        medium_times = event_rows.loc[
            event_rows["Compound"].eq(REFERENCE_COMPOUND).fillna(False), target
        ]
        for compound, rows in event_rows.groupby("Compound"):
            tyre_life = numeric_comparison(rows, "TyreLife_start", target)
            if tyre_life is not None:
                comparisons.append({**context, "compound": compound, **tyre_life})

            n, medium_n = len(rows), len(medium_times)
            if compound != REFERENCE_COMPOUND and min(n, medium_n) >= MIN_GROUP_ROWS:
                comparisons.append(
                    {
                        **context,
                        "feature": "Compound",
                        "compound": compound,
                        "reference_compound": REFERENCE_COMPOUND,
                        "n": n,
                        "reference_n": medium_n,
                        "statistic": "median_difference_s",
                        "value": rows[target].median() - medium_times.median(),
                    }
                )
    return pd.DataFrame(comparisons, columns=FEATURE_COLUMNS).astype({"reference_n": "Int64"})
