"""Charts: the numbers behind them match the forecast, and every chart draws."""

import numpy as np
import pytest

from mortgage_elasticity import diagnostics, visuals
from mortgage_elasticity.forecast import forecast

pytestmark = pytest.mark.filterwarnings("ignore")


def test_price_effect_matches_the_forecast(fit):
    seg = fit.prep.segs[0]
    s = forecast(fit, seg, -10).summary.set_index("version").loc["price_effect"]
    base, scn, long_run = visuals.price_effect(fit, seg, [-10.0, 0.0], draws=fit.n)
    mult = scn.sum(-1) / base.sum(-1)[:, None]
    assert np.median(mult[:, 0]) == pytest.approx(s["multiplier_p50"], rel=1e-9)
    assert np.median(long_run[:, 0]) == pytest.approx(s["long_run_multiplier_p50"], rel=1e-9)
    assert np.allclose(mult[:, 1], 1) and np.allclose(long_run[:, 1], 1)  # no change, no effect


def test_gallery_writes_every_chart(fit, sim, tmp_path):
    diagnostics.raw_move_check(fit.prep, fit.cfg).to_csv(tmp_path / "check_moves.csv", index=False)
    written = visuals.gallery(fit, tmp_path / "charts", margin_bps=180, outputs=tmp_path,
                              truth=sim[0] / "truth.json")
    names = {p.stem for p in written}
    assert {"response_curve", "share_history", "forecast_path_apps", "segment_move_heatmap",
            "competitor_response_3d", "phase_in_surface", "margin_tradeoff", "accuracy_groups",
            "check_moves"} <= names
    assert all(p.stat().st_size > 5000 for p in written)
