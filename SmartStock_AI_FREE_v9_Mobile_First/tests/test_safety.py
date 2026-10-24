from pathlib import Path
import sys
from datetime import date
import math
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from risk_engine import data_audit,risk_report,hypothetical_size,historical_stress,compare_market
from engine import clean_prices,indicators,trade_levels
from strict_validation import prediction_at,audit_history

DATA=Path(__file__).resolve().parents[1]/'demo_synthetic.csv'
def prices():
    return clean_prices(pd.read_csv(DATA,parse_dates=['Date'],index_col='Date'))

def test_data_quality_fresh_and_stale_and_bad_candles():
    p=prices().copy()
    today=p.index[-1].date()
    fresh=data_audit(p,as_of=today)
    assert fresh.age_weekdays==0
    assert fresh.status!='차단'
    stale=data_audit(p,as_of=(p.index[-1]+pd.Timedelta(days=15)).date())
    assert stale.status=='차단' and stale.age_weekdays>=5
    future=data_audit(p,as_of=(p.index[-1]-pd.Timedelta(days=5)).date())
    assert future.status=='차단'
    bad=p.copy()
    bad.loc[bad.index[-1],'High']=bad.Close.iloc[-1]*.5
    assert data_audit(bad,as_of=today).status=='차단'


def test_risk_math_and_not_equate_stop_to_guaranteed_loss():
    p=prices()
    levels=trade_levels(indicators(p))
    rr=risk_report(p,levels)
    assert rr.max_drawdown is not None and rr.max_drawdown<=0
    assert rr.daily_cvar95<=rr.daily_var95<=0.05
    assert rr.ann_volatility>0
    assert rr.stop_distance_pct>0
    assert rr.reward_risk>0
    sizing=hypothetical_size(entry=100,stop=94,portfolio=10000,risk_percent=1,max_allocation_percent=20)
    assert sizing['shares']==16
    assert sizing['planned_loss_to_stop']==96
    assert sizing['position_value']==1600
    assert sizing['planned_loss_to_stop']<=sizing['risk_budget']
    with pytest.raises(ValueError):hypothetical_size(100,110,10000,1)
    assert not historical_stress(p,(5,21,63)).empty


def test_market_comparison_is_aligned_and_not_future_aware():
    p=prices()
    q=p.copy()
    q['Close']=p.Close*1.15
    meta,out,norm=compare_market(p,q)
    assert len(out)==4
    assert math.isclose(meta['beta'],1,rel_tol=1e-6)
    assert math.isclose(meta['correlation'],1,rel_tol=1e-6)
    assert np.allclose(norm['종목'],norm['비교지수'])


def test_historical_predictor_cannot_see_future_bars():
    p=prices()
    t=950
    h=21
    earlier=prediction_at(p.iloc[:t+1],t,h)
    whole=prediction_at(p,t,h)
    assert earlier is not None and whole is not None
    assert earlier[1:]==whole[1:]
    assert abs(earlier[0]-whole[0])<1e-12
    # Changing all prices *after* the evaluation date cannot change its output.
    altered=p.copy()
    for name in ['Open','High','Low','Close']:
        altered.iloc[t+1:, altered.columns.get_loc(name)] *= 3.0
    after=prediction_at(altered,t,h)
    assert after is not None and abs(earlier[0]-after[0])<1e-12


def test_workforward_reliability_gates_are_descriptive():
    p=prices()
    audit=audit_history(p,21,.7,max_checks=8)
    assert audit.evaluations>=1
    assert len(audit.reliability_bins)==5
    assert len(audit.regime_reports)>=1
    assert audit.score_brier>=0 and audit.baseline_brier>=0
    assert audit.recommendation in ('판단 보류','조건부 매수 검토')
    assert sum(row['표본'] for row in audit.reliability_bins)==audit.evaluations
