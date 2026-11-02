"""Regression tests for defects reproduced in 10.8."""
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import elliott_waves
from engine import clean_prices,indicators
from market_data_guard import daily_market_state
from market_decision import _simulate,_features,_signals


def bars(n=100):
    d=pd.bdate_range('2026-05-01',periods=n)
    v=np.linspace(102,130,n)
    return pd.DataFrame({'Open':v,'High':v+2,'Low':v-2,'Close':v,'Volume':1e5},index=d)


def test_reject_close_above_high_before_stochastic_and_model():
    x=bars()
    x.iloc[55,x.columns.get_loc('Close')]=500
    with pytest.raises(ValueError,match='OHLC'):
        clean_prices(x)
    with pytest.raises(ValueError,match='OHLC'):
        indicators(x)


def test_reject_negative_and_missing_prices_without_silent_drop():
    x=bars();x.iloc[60,x.columns.get_loc('Low')]=-1
    with pytest.raises(ValueError):clean_prices(x)
    x=bars();x.iloc[60,x.columns.get_loc('Open')]=float('nan')
    with pytest.raises(ValueError):clean_prices(x)


def _fake_pivots(d,order,lookback):
    pts=[(20,207,'L'),(35,235,'H'),(45,209,'L'),(60,244,'H')]
    return [{'position':i,'date':str(d.index[i].date()),'time':d.index[i],'value':v,'kind':kind}
            for i,v,kind in pts]


def test_live_wave4_break_of_wave1_invalidates_existing_count(monkeypatch):
    x=bars(85)
    x['Low']=np.minimum(x['Low'],236)
    x['High']=np.maximum(x['High'],246)
    # Last observed low violates the wave-1 high 235
    x.iloc[-1,x.columns.get_loc('Low')]=201
    monkeypatch.setattr(elliott_waves,'_alternating_pivots',_fake_pivots)
    result=elliott_waves.estimate_wave(x,order=5)
    assert result['status']=='판정 보류'
    assert result['nodes']==[]
    assert result['invalidated_candidates']>=1
    assert '무효' in result['notes']


def test_wave4_live_candidate_stays_when_no_overlap(monkeypatch):
    x=bars(85)
    # last 24 bars have lows >235, and ordinary bars until pivot have no effect
    x.loc[x.index[61:],'Low']=236
    x.loc[x.index[61:],'High']=np.maximum(x.loc[x.index[61:],'High'],x.loc[x.index[61:],'Close'])
    monkeypatch.setattr(elliott_waves,'_alternating_pivots',_fake_pivots)
    result=elliott_waves.estimate_wave(x,order=5)
    assert result['status']=='후보'
    assert result['current']=='4파 조정 진행 후보'


def test_market_excludes_open_us_daily_candle():
    x=pd.DataFrame({'Open':[100.,105.],'High':[108.,110.],'Low':[98.,104.],
                    'Close':[105.,109.],'Volume':[100.,200.]},
                    index=pd.DatetimeIndex(['2026-10-30','2026-11-02']))
    intraday=datetime(2026,11,2,16,0,tzinfo=timezone.utc) # 11am New York
    filtered,msg=daily_market_state(x,'NVDA',now=intraday)
    assert len(filtered)==1 and '제외' in msg
    post=datetime(2026,11,2,23,0,tzinfo=timezone.utc)
    filtered,msg=daily_market_state(x,'NVDA',now=post)
    assert len(filtered)==2


def test_market_excludes_open_kr_daily_candle():
    x=pd.DataFrame({'Open':[100.,105.],'High':[108.,110.],'Low':[98.,104.],
                    'Close':[105.,109.],'Volume':[100.,200.]},
                    index=pd.DatetimeIndex(['2026-10-30','2026-11-02']))
    filtered,_=daily_market_state(x,'005930.KS',now=datetime(2026,11,2,5,0,tzinfo=timezone.utc))
    assert len(filtered)==1
    filtered,_=daily_market_state(x,'005930.KS',now=datetime(2026,11,2,8,0,tzinfo=timezone.utc))
    assert len(filtered)==2


def test_simulator_rearms_after_actual_exit_not_full_horizon():
    p=bars(120)
    f=_signals(_features(p))
    f.loc[:,['breakout','pullback','rebound']]=False
    for k in (73,75):f.iloc[k,f.columns.get_loc('breakout')]=True
    # Immediate next-bar stop outs in two separate trades
    for k in (74,76):p.iloc[k,p.columns.get_loc('Low')]=20
    deals=_simulate(p,f,'breakout',21,1.8)
    assert len(deals)==2
    assert [d['signal_i'] for d in deals]==[73,75]
    assert [d['exit_i'] for d in deals]==[74,76]


def test_unsettled_trade_not_counted_as_profit():
    p=bars(100)
    f=_signals(_features(p));f.loc[:,['breakout','pullback','rebound']]=False
    f.iloc[95,f.columns.get_loc('breakout')]=True
    assert _simulate(p,f,'breakout',21,1.8)==[]
