"""10.7: no hidden trade claims, stable history, conservative execution assumptions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from market_decision import (evaluate_market,example_position,_features,_signals,
                             _simulate,PROFILES)


def candles(n=280):
    idx=pd.bdate_range('2025-01-01',periods=n)
    close=100 + np.arange(n)*.11 + np.sin(np.arange(n)/6)*2
    return pd.DataFrame({'Open':close-.15,'High':close+1.5,'Low':close-1.5,
                         'Close':close, 'Volume':np.full(n,100000.)}, index=idx)


def test_five_stage_vocab_and_no_labeled_probability():
    p=candles();x=evaluate_market(p,score=.73,horizon=21)
    assert x.stage in {'적극 매수 검토','시험 매수 후보(모의매매 전용)','매수 대기','관망','위험 회피'}
    assert '확률 아님' in ' '.join(x.reasons)
    assert x.profile=='균형형' and x.risk_pct==.5 and x.max_alloc_pct==20
    assert len(x.stats)==3


def test_hard_block_synthetic_data_never_buy():
    p=candles();x=evaluate_market(p,score=.99,strict_recommendation='조건부 매수 검토',hypothetical=True)
    assert x.stage=='관망' and not x.verified
    assert x.entry is None


def test_hard_block_bad_data_never_buy():
    x=evaluate_market(candles(),score=.99,strict_recommendation='조건부 매수 검토',quality='차단')
    assert x.stage=='관망' and x.entry is None


def test_unverified_model_does_not_create_verified_buy():
    p=candles();p.iloc[-1,p.columns.get_loc('Volume')]=1_000_000
    p.iloc[-1,p.columns.get_loc('Close')]=float(p.Close.iloc[-2])+12
    p.iloc[-1,p.columns.get_loc('High')]=float(p.Close.iloc[-1])+1
    p.iloc[-1,p.columns.get_loc('Low')]=float(p.Close.iloc[-1])-1
    p.iloc[-1,p.columns.get_loc('Open')]=float(p.Close.iloc[-1])-1
    x=evaluate_market(p,score=.95,strict_recommendation='판단 보류')
    assert x.stage!='적극 매수 검토'
    assert x.verified is False or x.stage!='적극 매수 검토'


def test_leveraged_risk_limited():
    x=evaluate_market(candles(),symbol='TSLL',profile='적극형',score=.8)
    assert x.risk_pct<=.25 and x.max_alloc_pct<=10
    assert x.stage!='적극 매수 검토'


def test_strategies_use_past_only():
    p=candles(280)
    a=_signals(_features(p.iloc[:250]))
    altered=p.copy()
    altered.loc[altered.index[250]:,['Close','High','Low','Open']]*=5
    b=_signals(_features(altered)).loc[a.index]
    for col in ('breakout','pullback','rebound'):
        assert a[col].equals(b[col])


def test_cost_included_in_position_budget():
    p=candles();p.iloc[-1,p.columns.get_loc('Volume')]=1_000_000
    p.iloc[-1,p.columns.get_loc('Close')]=float(p.Close.iloc[-2])+12
    p.iloc[-1,p.columns.get_loc('High')]=float(p.Close.iloc[-1])+1
    p.iloc[-1,p.columns.get_loc('Low')]=float(p.Close.iloc[-1])-1
    p.iloc[-1,p.columns.get_loc('Open')]=float(p.Close.iloc[-1])-1
    x=evaluate_market(p,score=.73)
    if x.entry is not None:
        size=example_position(x,10000,.3)
        assert size['estimated_loss_if_stop_fills']<=size['risk_budget']+1e-8
        assert size['position_value']<=10000*x.max_alloc_pct/100+1e-8


def test_simulation_is_next_open_and_stop_first_on_same_bar():
    p=candles(120)
    f=_signals(_features(p))
    f.loc[:,['breakout','pullback','rebound']]=False
    t=72
    f.iloc[t,f.columns.get_loc('breakout')]=True
    close=float(p.Close.iloc[t]);atr=float(f.atr14.iloc[t]);
    p.iloc[t+1,p.columns.get_loc('Open')]=close*.99
    # Both above TP and below SL: stop wins
    p.iloc[t+1,p.columns.get_loc('High')]=close+10*atr
    p.iloc[t+1,p.columns.get_loc('Low')]=close-10*atr
    events=_simulate(p,f,'breakout',5,1.8)
    assert len(events)==1
    assert events[0]['entry']==pytest.approx(close*.99)
    assert events[0]['exit_type']=='손절(동시 터치 시 우선)'
    assert events[0]['net']<0


def test_no_future_outcome_in_tail():
    p=candles(100)
    f=_signals(_features(p))
    f.loc[:,['breakout','pullback','rebound']]=False
    f.iloc[97,f.columns.get_loc('breakout')]=True
    assert _simulate(p,f,'breakout',5,1.8)==[]


def test_invalid_inputs():
    p=candles()
    with pytest.raises(ValueError):evaluate_market(p,horizon=0)
    with pytest.raises(ValueError):evaluate_market(p,profile='최고수익')
    x=evaluate_market(p)
    with pytest.raises(ValueError):example_position(x,-1)


def test_quality_warning_not_same_as_hard_block():
    p=candles();p.iloc[-1,p.columns.get_loc('Volume')]=1_000_000
    p.iloc[-1,p.columns.get_loc('Close')]=float(p.Close.iloc[-2])+12
    p.iloc[-1,p.columns.get_loc('High')]=float(p.Close.iloc[-1])+1
    p.iloc[-1,p.columns.get_loc('Low')]=float(p.Close.iloc[-1])-1
    p.iloc[-1,p.columns.get_loc('Open')]=float(p.Close.iloc[-1])-1
    x=evaluate_market(p,score=.82,quality='주의')
    assert x.stage != '적극 매수 검토'
    assert x.stage in ('시험 매수 후보(모의매매 전용)','매수 대기','관망')
