"""Step 1-2: load the segment x period panel, derive share and diagnostics."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_elasticity.model import lag, softmin_rate

PANEL_COLS = [
    "segment", "period", "lodg_mkt", "lodg_us", "apps",
    "headline_cum_bps", "discretionary_cum_bps",
]
# optional: clean (default TRUE), other_changes, d_headline_bps / d_discretionary_bps (informational)
EVENT_COLS = ["segment", "period", "reason"]
_NONE = {"", "nan", "none", "n/a", "na", "-", "no"}  # other_changes values meaning "nothing else"
_TRUE, _FALSE = {"1", "1.0", "true", "t", "yes", "y"}, {"0", "0.0", "false", "f", "no", "n"}


def _rows(mask) -> list[int]:
    """First few offending rows as csv line numbers (header = line 1)."""
    return list(np.flatnonzero(np.asarray(mask))[:5] + 2)


def _read(path: Path, required: list[str], what: str, date_format: str | None,
          date_col: str = "period") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"segment": str})
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{what} is missing columns: {sorted(missing)}")
    df["segment"] = df["segment"].str.strip()
    if df["segment"].isna().any():
        raise ValueError(f"{what}: blank segment in rows {_rows(df['segment'].isna())}")
    if df["segment"].str.contains(r"[\[\]]").any():
        raise ValueError(f"{what}: segment names must not contain [ or ]")
    period = pd.to_datetime(df[date_col], format=date_format or "ISO8601", errors="coerce")
    if period.isna().any():
        raise ValueError(f"{what}: unparseable {date_col} in rows {_rows(period.isna())} "
                         "(set Settings.date_format, e.g. '%d/%m/%Y')")
    df[date_col] = period
    return df


def _bool(col: pd.Series, what: str) -> pd.Series:
    c = col.astype(str).str.strip().str.lower()
    bad = ~c.isin(_TRUE | _FALSE)
    if bad.any():
        raise ValueError(f"{what}: unrecognised {col.name} value in rows {_rows(bad)} "
                         "(use TRUE/FALSE, 1/0, yes/no)")
    return c.isin(_TRUE)


def load_panel(path: Path, date_format: str | None = None) -> pd.DataFrame:
    df = _read(path, PANEL_COLS, "panel", date_format)
    for c in PANEL_COLS[2:]:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            raise ValueError(f"panel: {c} blank or non-numeric in rows {_rows(v.isna())}")
        df[c] = v
    dup = df.duplicated(["segment", "period"])
    if dup.any():
        raise ValueError(f"panel: duplicate (segment, period) in rows {_rows(dup)}")
    if df.groupby("segment")["period"].apply(frozenset).nunique() > 1:
        raise ValueError("panel: segments cover different periods; the panel must be balanced")
    gaps = np.diff(np.sort(df["period"].unique())).astype("timedelta64[D]").astype(float)
    if len(gaps) and gaps.max() > 1.5 * np.median(gaps):
        warnings.warn(f"panel: irregular period spacing (largest gap {gaps.max():.0f} days)")
    df = df.sort_values(["segment", "period"]).reset_index(drop=True)
    df["share"] = df["lodg_us"] / df["lodg_mkt"]
    bad = (df["share"] <= 0) | (df["share"] >= 1)
    if bad.any():
        where = ", ".join(f"{r.segment}@{r.period.date()}" for r in df[bad].head(5).itertuples())
        raise ValueError(f"{bad.sum()} rows with share outside (0, 1) (e.g. {where}); "
                         "check segment alignment")
    df["y"] = np.log(df["share"] / (1 - df["share"]))
    df["conv"] = df["apps"] / df["lodg_us"]
    return df


def load_events(path: Path, segments=None, date_format: str | None = None) -> pd.DataFrame:
    ev = _read(path, EVENT_COLS, "events", date_format)
    ev["clean"] = _bool(ev["clean"], "events") if "clean" in ev else True
    # what else changed alongside the move (campaign, broker incentive, policy...): "" = nothing
    other = ev.get("other_changes", pd.Series("", index=ev.index)).fillna("").astype(str).str.strip()
    ev["other_changes"] = other.where(~other.str.lower().isin(_NONE), "")
    if segments is not None:
        unknown = sorted(set(ev["segment"]) - set(segments))
        if unknown:
            raise ValueError(f"events for segments not in the panel: {unknown}")
    if ev.duplicated(["segment", "period"]).any():
        warnings.warn("events: duplicate (segment, period) rows")
    return ev


def _per_period(log: pd.DataFrame, periods) -> pd.DataFrame:
    """Change log -> day-weighted average of the rate in force over each period (prev end, end].
    The first period is one median spacing long; before a lender's first change it is unknown."""
    periods = pd.DatetimeIndex(np.sort(np.unique(periods)))
    step = pd.Series(periods).diff().median() if len(periods) > 1 else pd.Timedelta(days=1)
    days = pd.date_range(periods[0] - step + pd.Timedelta(days=1), periods[-1], freq="D")
    which = periods[np.searchsorted(periods, days)]
    cols = ["rate_pct", "adj_bps", "eff_bps"]
    parts = []
    for (seg, lender), g in log.groupby(["segment", "lender"]):
        last = g.sort_values("effective_date").groupby("effective_date")[cols].last()
        daily = last.reindex(days, method="ffill")  # NaN before the first change
        off = np.isinf(daily["eff_bps"]).to_numpy()  # ineligible days
        daily.loc[off] = np.nan
        avg = daily.groupby(which).mean()
        avg.loc[pd.Series(off).groupby(which).all().to_numpy(), "eff_bps"] = np.inf
        avg = avg.rename_axis("period").reset_index()
        parts.append(avg.assign(segment=seg, lender=lender))
    return pd.concat(parts, ignore_index=True).dropna(subset=["eff_bps"])


def load_competitors(path: Path, exclude=(), periods=None,
                     date_format: str | None = None) -> pd.DataFrame:
    """Competitor rates per (period, segment, lender), or a change log with `effective_date`
    instead of `period` (converted with _per_period; needs the panel's `periods`).
    eff_bps = rate_pct * 100 - adj_bps  (adj_bps + = value to the customer)."""
    log = "period" not in pd.read_csv(path, nrows=0).columns
    date_col = "effective_date" if log else "period"
    c = _read(path, [date_col, "segment", "lender", "rate_pct"], "competitors", date_format,
              date_col)
    c["lender"] = c["lender"].astype(str).str.strip()
    c["rate_pct"] = pd.to_numeric(c["rate_pct"], errors="coerce")
    c["adj_bps"] = pd.to_numeric(c["adj_bps"], errors="coerce") if "adj_bps" in c else 0.0
    if c[["rate_pct", "adj_bps"]].isna().any().any():
        raise ValueError("competitors: blank or non-numeric rate_pct / adj_bps")
    if c["rate_pct"].median() > 100:
        raise ValueError("competitors: rate_pct looks like bps, expected percent (e.g. 6.19)")
    if c.duplicated([date_col, "segment", "lender"]).any():
        raise ValueError("competitors: duplicate (segment, period, lender) rows; collapse to the "
                         "lender's best effective product first")
    c["eff_bps"] = c["rate_pct"] * 100 - c["adj_bps"]
    if "eligible" in c:
        off = ~_bool(c["eligible"], "competitors")
        # +inf, not NaN: stops ffill carrying the old rate forward, and gets zero weight in r*
        c.loc[off, ["rate_pct", "adj_bps"]], c.loc[off, "eff_bps"] = np.nan, np.inf
    c = c[~c["lender"].isin(exclude)]
    if log:
        if periods is None:
            raise ValueError("competitors change log needs the panel periods")
        c = _per_period(c, periods)
    c = c.dropna(subset=["eff_bps"])
    return c[["period", "segment", "lender", "rate_pct", "adj_bps", "eff_bps"]]


def _wide(comp: pd.DataFrame, seg, periods, ffill: int, value: str = "eff_bps") -> pd.DataFrame:
    w = comp[comp["segment"] == seg].pivot(index="period", columns="lender", values=value)
    w = w.reindex(w.index.union(periods)).ffill(limit=ffill).reindex(periods)
    return w.where(~np.isinf(w))  # ineligible -> absent


def add_benchmark(df: pd.DataFrame, comp: pd.DataFrame, T: pd.Series, ffill: int,
                  min_lenders: int = 3) -> pd.DataFrame:
    """Adds `r_star_bps`: soft-min of competitors' effective rates (level, bps), per segment."""
    out = df.copy()
    out["r_star_bps"] = np.nan
    for seg, g in out.groupby("segment", sort=False):
        if not (comp["segment"] == seg).any():
            raise ValueError(f"competitors: no rows for panel segment {seg}")
        w = _wide(comp, seg, g["period"], ffill)
        n = w.notna().sum(axis=1)
        if (n == 0).any():
            gap = [str(p.date()) for p in w.index[n == 0][:5]]
            raise ValueError(f"competitors: no lender rates for {seg} in periods {gap} "
                             "(raise comp_ffill_periods?)")
        if n.min() < min_lenders:
            warnings.warn(f"{seg}: only {n.min()} competitors in some periods")
        if n.nunique() > 1:
            warnings.warn(f"{seg}: number of competitors changes over time ({n.min()}-{n.max()}); "
                          "entry/exit shifts r* by T*log(N) without any rate moving")
        if (w.min(axis=1) < w.median(axis=1) - 100).any():
            warnings.warn(f"{seg}: a competitor rate sits >100bp below the median "
                          "(teaser/outlier?)")
        out.loc[g.index, "r_star_bps"] = softmin_rate(w.to_numpy(), T[seg])
    return out


def benchmark_snapshot(comp: pd.DataFrame, T: pd.Series, periods, ffill: int) -> pd.DataFrame:
    """Last-period competitor table per segment, with each lender's weight in r*."""
    last, parts = pd.DatetimeIndex([max(periods)]), []
    for seg in T.index:
        row = {v: _wide(comp, seg, periods, ffill, v).reindex(last).iloc[0]
               for v in ["rate_pct", "adj_bps", "eff_bps"]}
        t = pd.DataFrame(row).dropna(subset=["eff_bps"]).rename_axis("lender").reset_index()
        e = t["eff_bps"].to_numpy()
        wt = (np.exp(-(e - e.min()) / T[seg]) if T[seg] > 0 else (e == e.min()).astype(float))
        t["weight"] = wt / wt.sum()
        t["r_star_pct"] = softmin_rate(e[None, :], T[seg])[0] / 100
        parts.append(t.assign(segment=seg))
    return pd.concat(parts, ignore_index=True)[
        ["segment", "lender", "rate_pct", "adj_bps", "eff_bps", "weight", "r_star_pct"]]


def add_perceived_price(df: pd.DataFrame, kappa: float, hH: float, hD: float,
                        demean: bool = False) -> pd.DataFrame:
    """x = lagged headline + kappa * lagged discretionary.
    Gap model: if `r_star_bps` is present, the lagged move in the competitor benchmark is
    subtracted. Move model (no r_star): `demean` strips the period mean across segments, i.e.
    moves made everywhere at once (RBA pass-through, and our own bank-wide moves)."""
    out = df.copy()
    out["x"] = np.nan
    for _, g in out.groupby("segment", sort=False):
        x = lag(g["headline_cum_bps"], hH) + kappa * lag(g["discretionary_cum_bps"], hD)
        if "r_star_bps" in g:
            x = x - lag(g["r_star_bps"] - g["r_star_bps"].iloc[0], hH)
        out.loc[g.index, "x"] = x
    if demean and "r_star_bps" not in out:
        out["x"] -= out.groupby("period")["x"].transform("mean")
    return out


def baseline_share(df: pd.DataFrame, window: int) -> pd.Series:
    """s0 per segment: pooled share over the trailing `window` periods."""
    tail = df.groupby("segment", sort=False).tail(window)
    agg = tail.groupby("segment")[["lodg_us", "lodg_mkt"]].sum()
    return (agg["lodg_us"] / agg["lodg_mkt"]).rename("s0")


def conversion_report(df: pd.DataFrame) -> pd.DataFrame:
    """Is apps/lodgements stable? A large trend or CV means the lodgement proxy needs adjusting."""
    rows = []
    for seg, g in df.groupby("segment"):
        if len(g) < 3 or (g["lodg_us"] <= 0).any():
            rows.append({"segment": seg})  # too short / undefined: left blank
            continue
        t = np.arange(len(g))
        slope = np.polyfit(t, g["conv"].to_numpy(), 1)[0]
        mean = g["conv"].mean()
        rows.append({
            "segment": seg,
            "conv_mean": mean,
            "conv_cv": g["conv"].std() / mean,
            "conv_trend_per_period_pct": slope / mean,
        })
    return pd.DataFrame(rows)
