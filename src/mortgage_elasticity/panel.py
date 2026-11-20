"""Load and validate the inputs, build the competitor benchmark and the unlagged price gap."""

import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_elasticity import runlog
from mortgage_elasticity.model import lag, softmin_rate

PANEL_COLS = ["segment", "period", "lodg_mkt", "lodg_us", "apps", "headline_cum_bps"]
COUNT_COLS = ["lodg_mkt", "lodg_us", "apps"]
EVENT_COLS = ["date", "segments", "type", "description"]
EVENT_TYPES = ("price", "other")
PURPOSES, REPAYMENTS = ("OO", "INV"), ("PI", "IO")
_TRUE, _FALSE = {"1", "1.0", "true", "t", "yes", "y"}, {"0", "0.0", "false", "f", "no", "n"}


def _rows(mask) -> list[int]:
    """First few offending rows as csv line numbers (header = line 1)."""
    return list(np.flatnonzero(np.asarray(mask))[:5] + 2)


def _read(path: Path, required: list[str], what: str, date_format: str | None,
          date_col: str = "period") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"segment": str, "segments": str})
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{what} is missing columns: {sorted(missing)}")
    if "segment" in df:
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


def load_panel(path: Path, date_format: str | None = None,
               covariates: tuple[str, ...] = ()) -> pd.DataFrame:
    df = _read(path, PANEL_COLS + list(covariates), "panel", date_format)
    if "discretionary_cum_bps" not in df:
        runlog.log("panel: no discretionary_cum_bps column; discretionary discount changes treated as "
            "zero (the model sees carded moves only)")
        df["discretionary_cum_bps"] = 0.0
    for c in PANEL_COLS[2:] + ["discretionary_cum_bps", *covariates]:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            raise ValueError(f"panel: {c} blank or non-numeric in rows {_rows(v.isna())}")
        df[c] = v
    for c in COUNT_COLS:
        bad = (df[c] < 0) | (df[c] != df[c].round())
        if bad.any():
            raise ValueError(f"panel: {c} negative or not a whole number in rows {_rows(bad)}")
        df[c] = df[c].astype(int)
    dup = df.duplicated(["segment", "period"])
    if dup.any():
        raise ValueError(f"panel: duplicate (segment, period) in rows {_rows(dup)}")
    bad = df["lodg_us"] > df["lodg_mkt"]
    if bad.any():
        raise ValueError(f"panel: lodg_us > lodg_mkt in rows {_rows(bad)}; check segment "
                         "alignment")
    if df.groupby("segment")["period"].apply(frozenset).nunique() > 1:
        raise ValueError("panel: segments cover different periods; the panel must be balanced")
    gaps = np.diff(np.sort(df["period"].unique())).astype("timedelta64[D]").astype(float)
    if len(gaps) and gaps.max() > 1.5 * np.median(gaps):
        warnings.warn(f"panel: irregular period spacing (largest gap {gaps.max():.0f} days)")
    df = df.sort_values(["segment", "period"]).reset_index(drop=True)
    df["share"] = (df["lodg_us"] / df["lodg_mkt"]).where(df["lodg_mkt"] > 0)  # NaN if no market
    return df


def load_events(path: Path, segments, date_format: str | None = None) -> pd.DataFrame:
    """Event log: one row per change. `segments` is ALL (bank-wide) or a ;-separated list.
    `type` is price (an own price move; its effect is carried by x) or other (campaign, policy,
    commission...; gets a level jump). Returns `segments` as a tuple of panel segments."""
    ev = _read(path, EVENT_COLS, "events", date_format, "date")
    segments = list(segments)
    ev["type"] = ev["type"].astype(str).str.strip().str.lower()
    bad = ~ev["type"].isin(EVENT_TYPES)
    if bad.any():
        raise ValueError(f"events: type must be price or other, in rows {_rows(bad)}")
    ev["description"] = ev["description"].fillna("").astype(str).str.strip()
    parsed = []
    for raw in ev["segments"].fillna("").astype(str):
        parts = [p.strip() for p in raw.replace("|", ";").split(";") if p.strip()]
        if not parts:
            raise ValueError("events: blank segments (use ALL for a bank-wide change)")
        if [p.upper() for p in parts] == ["ALL"]:
            parts = segments
        unknown = sorted(set(parts) - set(segments))
        if unknown:
            raise ValueError(f"events for segments not in the panel: {unknown}")
        parsed.append(tuple(s for s in segments if s in parts))
    ev["all"] = [len(p) == len(segments) for p in parsed]
    ev["segments"] = parsed
    dup = ev.duplicated(["date", "segments", "type", "description"])
    if dup.any():
        raise ValueError(f"events: duplicate rows {_rows(dup)}")
    ev = ev.sort_values("date", kind="stable").reset_index(drop=True)
    return ev[["date", "segments", "all", "type", "description"]]


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
                  min_lenders: int = 3, quiet: bool = False) -> pd.DataFrame:
    """Adds `r_star_bps`: soft-min of competitors' effective rates (level, bps), per segment.
    quiet=True skips the data warnings (for rebuilds at a new T; the data hasn't changed)."""
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
        if quiet:
            pass
        elif n.min() < min_lenders:
            warnings.warn(f"{seg}: only {n.min()} competitors in some periods")
        if not quiet and n.nunique() > 1:
            warnings.warn(f"{seg}: number of competitors changes over time ({n.min()}-{n.max()}); "
                          "entry/exit shifts r* by T*log(N) without any rate moving")
        if not quiet and (w.min(axis=1) < w.median(axis=1) - 100).any():
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


def add_gap(df: pd.DataFrame, kappa: float) -> pd.DataFrame:
    """Adds `gap_raw`, the unlagged price gap: headline + kappa * discretionary, minus the move
    in the competitor benchmark since the first period (no `r_star_bps` column = own move only)."""
    out = df.copy()
    out["gap_raw"] = out["headline_cum_bps"] + kappa * out["discretionary_cum_bps"]
    if "r_star_bps" in out:
        out["gap_raw"] -= out["r_star_bps"] - out.groupby("segment")["r_star_bps"].transform("first")
    return out


def add_perceived_price(df: pd.DataFrame, half_life: float) -> pd.DataFrame:
    """Adds `x`: gap_raw phased in with one half-life (model.lag), per segment."""
    out = df.copy()
    out["x"] = out.groupby("segment", sort=False)["gap_raw"].transform(
        lambda v: lag(v.to_numpy(), half_life))
    return out


def segment_dims(segments, bands: tuple[str, ...]) -> pd.DataFrame:
    """Split PURPOSE_REPAYMENT_LVRBAND segment names (e.g. OO_PI_70-80) into their dimensions."""
    rows = []
    for seg in segments:
        parts = str(seg).split("_")
        if (len(parts) != 3 or parts[0] not in PURPOSES or parts[1] not in REPAYMENTS
                or parts[2] not in bands):
            raise ValueError(
                f"panel: segment '{seg}' is not PURPOSE_REPAYMENT_LVRBAND with purpose in "
                f"{PURPOSES}, repayment in {REPAYMENTS} and LVR band in {tuple(bands)} "
                "(e.g. OO_PI_70-80)")
        rows.append({"segment": seg, "purpose": parts[0], "repayment": parts[1],
                     "lvr_band": parts[2]})
    return pd.DataFrame(rows).set_index("segment")


def conversion_report(df: pd.DataFrame, window: int = 13) -> pd.DataFrame:
    """Conversion ratio (trailing-window apps / lodgements) and whether it is stable. A large
    trend or CV means the lodgement proxy needs adjusting."""
    rows = []
    for seg, g in df.groupby("segment", sort=False):
        tail = g.tail(window)
        row = {"segment": seg,
               "conversion": tail["apps"].sum() / tail["lodg_us"].sum()
               if tail["lodg_us"].sum() > 0 else np.nan}
        if len(g) >= 3 and (g["lodg_us"] > 0).all():  # else too short / undefined: left blank
            conv = (g["apps"] / g["lodg_us"]).to_numpy()
            row["conv_cv"] = conv.std(ddof=1) / conv.mean()
            row["conv_trend_per_period_pct"] = np.polyfit(np.arange(len(g)), conv, 1)[0] / conv.mean()
        rows.append(row)
    return pd.DataFrame(rows, columns=["segment", "conversion", "conv_cv",
                                       "conv_trend_per_period_pct"])


@dataclass(frozen=True)
class Prepared:
    """Model-ready arrays, periods x segments. `df` is the same data in long form."""
    df: pd.DataFrame
    segs: list
    periods: pd.DatetimeIndex
    dims: pd.DataFrame  # purpose / repayment / lvr_band per segment
    M: np.ndarray  # market lodgements
    L: np.ndarray  # our lodgements
    obs: np.ndarray  # in the likelihood (market lodgements > 0)
    gap_raw: np.ndarray  # unlagged price gap, bp
    s0: np.ndarray  # pooled share over the whole panel, per segment
    events: pd.DataFrame  # event log with `t`, the panel week each event lands in
    comp: pd.DataFrame | None
    T: pd.Series | None  # soft-min temperature per segment, bp
    Z: np.ndarray | None  # standardised covariates, periods x segments x covariates
    kappa: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.M.shape

    def wide(self, col: str) -> np.ndarray:
        return self.df.pivot(index="period", columns="segment", values=col)[self.segs].to_numpy()

    def moves(self, kind: str = "price") -> pd.DataFrame:
        return self.events[self.events["type"] == kind]

    def with_T(self, T: pd.Series, cfg) -> "Prepared":
        """Same data with the benchmark and the gap rebuilt at another temperature."""
        df = add_benchmark(self.df, self.comp, T, cfg.comp_ffill_periods, cfg.min_lenders, quiet=True)
        df = add_gap(df, self.kappa)
        return replace(self, df=df, T=T, gap_raw=_wide_of(df, self.segs, "gap_raw"))

    def with_kappa(self, kappa: float) -> "Prepared":
        df = add_gap(self.df, kappa)
        return replace(self, df=df, kappa=kappa, gap_raw=_wide_of(df, self.segs, "gap_raw"))


def _wide_of(df: pd.DataFrame, segs, col: str) -> np.ndarray:
    return df.pivot(index="period", columns="segment", values=col)[segs].to_numpy()


def prepare(df: pd.DataFrame, events: pd.DataFrame, comp: pd.DataFrame | None, T: pd.Series | None,
            cfg) -> Prepared:
    segs = list(dict.fromkeys(df["segment"]))
    periods = pd.DatetimeIndex(np.sort(df["period"].unique()))
    if comp is not None:
        df = add_benchmark(df, comp, T, cfg.comp_ffill_periods, cfg.min_lenders)
    else:
        warnings.warn("no competitor file: the price gap is our own move only, and competitor "
                      "moves are invisible to the model")
    df = add_gap(df, cfg.kappa)
    M, L = _wide_of(df, segs, "lodg_mkt"), _wide_of(df, segs, "lodg_us")
    obs = M > 0
    if (~obs).any():
        runlog.log(f"panel: {int((~obs).sum())} segment-weeks with zero market lodgements left "
                   "out of the likelihood")
    if (L.sum(0) == 0).any():
        raise ValueError("panel: no own lodgements at all in "
                         f"{[s for s, z in zip(segs, L.sum(0) == 0) if z]}")
    t = np.searchsorted(periods, events["date"].to_numpy(), side="left")
    if (t >= len(periods)).any():
        warnings.warn(f"events: {int((t >= len(periods)).sum())} dated after the last period, "
                      "dropped")
    events = events.assign(t=t)[t < len(periods)].reset_index(drop=True)
    events["period"] = periods[events["t"].to_numpy()]
    Z = None
    if cfg.covariates:
        Z = np.stack([_wide_of(df, segs, c) for c in cfg.covariates], axis=-1)
        Z = (Z - Z.mean((0, 1))) / Z.std((0, 1))
    return Prepared(df=df, segs=segs, periods=periods, dims=segment_dims(segs, cfg.lvr_bands),
                    M=M, L=L, obs=obs, gap_raw=_wide_of(df, segs, "gap_raw"),
                    s0=L.sum(0) / M.sum(0), events=events, comp=comp, T=T, Z=Z, kappa=cfg.kappa)
