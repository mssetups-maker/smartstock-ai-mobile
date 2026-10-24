import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import clean_prices,indicators,make_features,analyze,HORIZON,trade_levels,swing_levels

DATA=Path(__file__).resolve().parents[1]/'demo_synthetic.csv'

def fixture_prices():
    return clean_prices(pd.read_csv(DATA,parse_dates=['Date'],index_col='Date'))

def test_features_do_not_change_when_future_rows_added():
    df=fixture_prices()
    earlier=make_features(indicators(df.iloc[:-80]))
    full=make_features(indicators(df))
    cols=list(earlier.columns)
    at=earlier.index[-1]
    np.testing.assert_allclose(earlier.loc[at,cols],full.loc[at,cols],rtol=1e-11,equal_nan=True)

def test_model_outputs_and_one_month_path():
    x=fixture_prices()
    out=analyze(indicators(x))
    assert 0<=out.probability<=1
    assert out.test_cases>12
    assert out.train_cases>400
    assert out.paths is not None
    assert len(out.paths['median'])==HORIZON+1
    assert np.all(out.paths['low']<=out.paths['median']+1e-10)
    assert np.all(out.paths['median']<=out.paths['high']+1e-10)
    assert out.levels['stop']<out.levels['entry']<out.levels['tp1']<out.levels['tp2']

def test_recent_missing_outcomes_not_labeled_false():
    from engine import make_features
    d=indicators(fixture_prices())
    futures=d['Close'].shift(-HORIZON)/d['Close']-1
    assert futures.tail(HORIZON).isna().all()
    assert pd.notna(make_features(d).iloc[-1]).all()

def test_swing_pivots_require_confirmation_bars():
    x=fixture_prices()
    supp,res=swing_levels(x,order=4)
    assert all(p[0] <= len(x)-5 for p in supp+res)
