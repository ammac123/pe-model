"""Excel export: the workbook's live formulas reproduce the Python price-effect forecast."""

import formulas
import numpy as np
import openpyxl
import pytest

from mortgage_elasticity import report
from mortgage_elasticity.excel import export_excel
from mortgage_elasticity.forecast import benchmark_shift, forecast

pytestmark = pytest.mark.filterwarnings("ignore")


def _calculate(path, tmp_path):
    """Evaluate every formula in the workbook (openpyxl only stores them)."""
    model = formulas.ExcelModel().loads(str(path)).finish()
    model.calculate()
    model.write(dirpath=str(tmp_path / "calc"))
    return openpyxl.load_workbook(next((tmp_path / "calc").glob("*.[xX][lL][sS][xX]")),
                                  data_only=True)


def _scenario(wb) -> dict:
    ws = wb[next(n for n in wb.sheetnames if n.lower() == "scenario")]
    head = [c.value for c in ws[2]]
    return {row[0]: dict(zip(head, row)) for row in ws.iter_rows(min_row=3, values_only=True)
            if row[0]}


def test_workbook_matches_the_python_forecast(fit, tmp_path):
    segs = fit.prep.segs
    lender = fit.prep.comp.loc[fit.prep.comp["segment"] == segs[1], "lender"].iloc[0]
    path = export_excel(fit, tmp_path / "elasticity.xlsx")
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["segments", "benchmark", "scenario"]
    assert "80% range from sensitivity only" in wb["scenario"]["I1"].value
    # scenario: -10bp headline in segment 0; +20bp less discount and one competitor 15bp cheaper
    # in segment 1 (typed in as that lender's new rate)
    sc, bm = wb["scenario"], wb["benchmark"]
    sc["B3"], sc["C4"] = -10, 20
    for row in bm.iter_rows(min_row=2):
        if row[0].value == segs[1] and row[1].value == lender:
            row[6].value = row[2].value - 0.15
    wb.save(path)
    got = _scenario(_calculate(path, tmp_path))

    shift = benchmark_shift(fit, {(segs[1], lender): -15})[1]
    assert shift < 0
    excel = {"settled": True, "median_h": True}  # what the workbook assumes
    cases = [(segs[0], forecast(fit, segs[0], -10, **excel), -10.0),
             (segs[1], forecast(fit, segs[1], 0, 20, {(segs[1], lender): -15}, **excel),
              fit.prep.kappa * 20 - shift)]
    for seg, f, dx in cases:
        want = f.summary.set_index("version").loc["price_effect"]
        row = got[seg]
        assert row["dx_bps"] == pytest.approx(dx, abs=1e-6)
        # beta p10 gives the smallest effect: the low end for a cut, the high end for a rise
        ends = ("p10", "p50", "p90") if dx < 0 else ("p90", "p50", "p10")
        for excel_q, python_q in zip(("p10", "p50", "p90"), ends):
            assert row[f"multiplier_{excel_q}"] == pytest.approx(want[f"multiplier_{python_q}"],
                                                                rel=0.005)
        assert row["long_run_p50"] == pytest.approx(want["long_run_multiplier_p50"], rel=0.005)
    idle = got[segs[2]]
    assert idle["multiplier_p50"] == pytest.approx(1.0) and idle["dx_bps"] == pytest.approx(0.0)


def test_segments_sheet_carries_the_fit(fit, tmp_path):
    wb = openpyxl.load_workbook(export_excel(fit, tmp_path / "e.xlsx"))
    rows = list(wb["segments"].iter_rows(values_only=True))
    assert rows[0] == ("segment", "share_now", "beta_p10", "beta_p50", "beta_p90", "T_bps",
                       "h_p50", "conversion", "prior_share")
    seg = report.segments_table(fit)
    assert np.allclose([r[3] for r in rows[1:]], seg["beta_p50"])
    assert np.allclose([r[6] for r in rows[1:]], np.median(fit.draws["h"]))
