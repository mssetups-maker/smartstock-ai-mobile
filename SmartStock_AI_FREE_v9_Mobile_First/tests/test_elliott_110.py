"""11.0: Real-time ABC integrity, alternative display and cautious targets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import elliott_waves as ew


def pivot(i,price,kind):
    return {'position':i,'date':str(pd.Timestamp('2026-01-01')+pd.Timedelta(days=i))[:10],
            'time':pd.Timestamp('2026-01-01')+pd.Timedelta(days=i),
            'value':float(price),'kind':kind}


def candles(n=105):
    idx=pd.date_range('2026-01-01',periods=n,freq='D')
    close=np.full(n,100.)
    return pd.DataFrame({'Open':close,'High':close+3,'Low':close-3,
                         'Close':close,'Volume':np.full(n,1e6)},index=idx)


DOWN=[pivot(12,130,'H'),pivot(22,90,'L'),pivot(32,122,'H'),
      pivot(42,68,'L'),pivot(52,85,'H'),pivot(62,60,'L')]


def test_directional_abc_pivots_enforced():
    a=pivot(72,117,'H');b=pivot(82,79,'L');c=pivot(92,125,'H')
    assert ew._correction_valid(DOWN,[a],-1,.3)
    assert ew._correction_valid(DOWN,[a,b],-1,.3)
    assert ew._correction_valid(DOWN,[a,b,c],-1,.3)
    assert not ew._correction_valid(DOWN,[a,b,pivot(92,112,'H')],-1,.3)
    assert not ew._correction_valid(DOWN,[a,pivot(82,50,'L')],-1,.3)


def test_abc_not_mislabeled_as_verified_zigzag(monkeypatch):
    d=candles(100)
    p=DOWN+[pivot(72,117,'H'),pivot(82,79,'L')]
    monkeypatch.setattr(ew,'_alternating_pivots',lambda *a,**k:p)
    z=ew.estimate_wave(d,order=5)
    assert z['status']=='후보'
    assert z['nodes'][-1]['파동']=='B'
    assert '5-3-5' in z['notes']
    assert z['current'].startswith('C파')
    assert z['projection']['zone'][0]>79
    assert z['projection']['zone'][1]>z['projection']['zone'][0]


def test_abc_breach_after_b_forces_recount(monkeypatch):
    d=candles(100)
    d.iloc[-1,d.columns.get_loc('Low')]=75 # after B (79) -> prospective C upward invalid
    monkeypatch.setattr(ew,'_alternating_pivots',lambda *a,**k:DOWN+[pivot(72,117,'H'),pivot(82,79,'L')])
    z=ew.estimate_wave(d,order=5)
    assert z['status']=='판정 보류'
    assert z['nodes']==[]
    assert z['recount_required']
    assert 'C파' in z['notes']


def test_abc_b_retrace_through_wave5_invalidates_simple_zigzag(monkeypatch):
    d=candles(95)
    d.iloc[-1,d.columns.get_loc('Low')]=55 # hypothetical B following A breaks wave 5
    monkeypatch.setattr(ew,'_alternating_pivots',lambda *a,**k:DOWN+[pivot(72,117,'H')])
    z=ew.estimate_wave(d,order=5)
    assert z['status']=='판정 보류'
    assert z['recount_required']
    assert 'B파' in z['notes']


def test_old_partial_counts_not_presented_as_current(monkeypatch):
    d=candles(100)
    # A potential 0-3 count followed by a newer contradictory confirmed swing
    p=DOWN[:4]+[pivot(52,200,'H')]
    monkeypatch.setattr(ew,'_alternating_pivots',lambda *a,**k:p)
    z=ew.estimate_wave(d,order=5)
    assert z['status']=='판정 보류'


def test_completed_c_then_stale_not_claimed_current(monkeypatch):
    d=candles(110)
    p=DOWN+[pivot(72,117,'H'),pivot(82,79,'L'),pivot(92,125,'H')]
    monkeypatch.setattr(ew,'_alternating_pivots',lambda *a,**k:p)
    z=ew.estimate_wave(d,order=5)
    assert z['status']=='판정 보류' # C was 17 bars ago, not current


def test_full_abc_strictly_separates_end_and_pending(monkeypatch):
    d=candles(100)
    p=DOWN+[pivot(72,117,'H'),pivot(82,79,'L'),pivot(92,125,'H')]
    monkeypatch.setattr(ew,'_alternating_pivots',lambda *a,**k:p)
    z=ew.estimate_wave(d,order=5)
    assert z['status']=='후보'
    assert z['nodes'][-1]['파동']=='C'
    assert z['current']=='ABC 완료 가능 · 새 구조 확인 필요'
    assert not z['projection']['zone']


def test_fig_renders_single_scenario_abc_different_color():
    from plotly.subplots import make_subplots
    p=DOWN+[pivot(72,117,'H'),pivot(82,79,'L')]
    z=ew._result_for_candidate(DOWN,p[6:],-1,p,5,[])
    f=make_subplots(rows=1,cols=1)
    ew.attach_wave_traces(f,z,latest=(pd.Timestamp('2026-05-01'),100.0))
    assert len(f.data)==3 # 0-5, A-B, current observed unconfirmed
    assert f.data[0].line.color!=f.data[1].line.color
    assert all('W' not in str(x) for trace in f.data for x in trace.text or [])


def test_consensus_is_not_claimed_probability(monkeypatch):
    def fake_estimate(d,order=5):
        return {'status':'판정 보류','phase':None,'current':'파동 번호 미확정','confirmed_through':None}
    monkeypatch.setattr(ew,'estimate_wave',fake_estimate)
    z=ew.wave_sensitivity_agreement(candles(),orders=(3,5,8))
    assert z['agreement'] is False
    assert len(z['rows'])==3
    assert '확률' in z['notes']
