import numpy as np
import pandas as pd

from reconciliation.base import apply_reconciliation_method
from reconciliation.bottom_up import reconcile_bottom_up
from reconciliation.metrics import calculate_hic, calculate_ocv, calculate_rd
from reconciliation.mint import reconcile_mint
from reconciliation.ols import reconcile_ols
from reconciliation.summing_matrix import build_summing_matrix
from reconciliation.top_down import reconcile_top_down
from reconciliation.wls import reconcile_wls


def test_bottom_up_returns_station_sum():
    y = reconcile_bottom_up(np.array([10.0, 20.0, 30.0]))
    assert y[0] == 60.0
    assert np.allclose(y[1:], [10.0, 20.0, 30.0])


def test_top_down_returns_coherent_units():
    y = reconcile_top_down(100.0, np.array([0.2, 0.3, 0.5]))
    assert np.isclose(y[0], np.sum(y[1:]))
    assert np.allclose(y[1:], [20.0, 30.0, 50.0])


def test_ols_returns_coherent_forecast():
    S = build_summing_matrix(2)
    y = reconcile_ols(np.array([100.0, 40.0, 50.0]), S)
    assert np.isclose(y[0], np.sum(y[1:]))


def test_wls_returns_coherent_forecast():
    S = build_summing_matrix(2)
    y = reconcile_wls(np.array([100.0, 40.0, 50.0]), S, np.ones(3))
    assert np.isclose(y[0], np.sum(y[1:]))


def test_mint_handles_nearly_singular_covariance():
    S = build_summing_matrix(2)
    cov = np.array([[1.0, 0.999999, 0.0], [0.999999, 1.0, 0.0], [0.0, 0.0, 1e-12]])
    y = reconcile_mint(np.array([100.0, 40.0, 50.0]), S, cov)
    assert np.isclose(y[0], np.sum(y[1:]))


def test_apply_reconciliation_method_ols_table():
    df = pd.DataFrame(
        {
            "N_chp_pred": [100.0],
            "N_1_pred": [40.0],
            "N_2_pred": [50.0],
            "N_chp_true": [95.0],
            "N_1_true": [45.0],
            "N_2_true": [50.0],
            "N_min_1": [0.0],
            "N_min_2": [0.0],
            "N_max_1": [100.0],
            "N_max_2": [100.0],
        }
    )
    out = apply_reconciliation_method(df, "ols", unit_ids=["B1", "B2"])
    assert np.isclose(out.loc[0, "N_chp_pred"], out.loc[0, ["N_1_pred", "N_2_pred"]].sum())


def test_hic_zero_for_coherent_forecast():
    hic = calculate_hic(pd.Series([30.0]), pd.DataFrame({"a": [10.0], "b": [20.0]}))
    assert hic["HIC_count_total"] == 0


def test_ocv_counts_min_and_max_violations():
    ocv = calculate_ocv(
        pd.DataFrame({"a": [5.0], "b": [25.0]}),
        pd.DataFrame({"a": [10.0], "b": [0.0]}),
        pd.DataFrame({"a": [20.0], "b": [20.0]}),
    )
    assert ocv["OCV_min_total"] == 1
    assert ocv["OCV_max_total"] == 1
    assert ocv["OCV_total"] == 2


def test_rd_uses_article_formula():
    rd = calculate_rd(pd.Series([90.0]), pd.Series([100.0]))
    assert np.isclose(rd["RD_signed_percent"], 10.0)
