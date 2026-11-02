"""10.11 regression: Elliott low-family candidate ≠ oracle bottom, prospective audit."""
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pytest
from wave_bottoms import detect_bottoms
from wave_bottom_audit import update_bottom_audit


def candles(points, step=7, pre=60):
    y=[]
    for a,b in zip(points,points[1:]):
        y.extend(np.linspace(a,b,step,endpoint=False))
    y.append(points[-1])
    prefix=[points[0]*(1+.005*np.sin(i/6)) for i in range(pre)]
    y=np.array(prefix+y, dtype=float)
    return pd.DataFrame({'Open':y,'High':y*1.004,'Low':y*.996,'Close':y,
                         'Volume':np.full(len(y),1000)},
                        index=pd.date_range('2025-01-01',periods=len(y),freq='D'))


def kinds(frame):
    return {c['kind']: c for c in detect_bottoms(frame,order=3)['candidates']}


def test_wave_2_potential_and_invalidation():
    observed=candles([100,145,118,122],9)
    r=kinds(observed)
    assert '2파 저점' in r
    w=r['2파 저점']
    assert w['zone_low'] < w['observed_low'] < w['zone_high']
    assert w['invalidation'] < w['observed_low']
    assert w['status'] in ('관찰 구간','초기 반등 관찰','후행 저점 확인')
    broken=candles([100,145,94,96],9)
    assert '2파 저점' not in kinds(broken)


def test_c_wave_zigzag_candidate():
    r=kinds(candles([160,100,132,82,86],7))
    assert 'C파 저점' in r
    c=r['C파 저점']
    assert c['reevaluate'] < c['observed_low']
    assert c['invalidation'] is None  # Fibonacci levels are not absolute rules.
    assert any('내부 구조 미확인' in z for z in c['reasons'])


def test_z_only_structural_skeleton():
    r=kinds(candles([200,161,181,137,154,105,110],7))
    assert 'Z파 저점' in r
    z=r['Z파 저점']
    assert len(z['anchors']) == 5
    assert any('내부 ABC 확인 없이' in x for x in z['reasons'])
    assert z['research_only']


def test_no_old_bottom_as_new_candidate():
    frame=candles([160,100,132,82,86],7)
    far_later=pd.DataFrame({'Open':[86.0]*45,'High':[86.7]*45,
            'Low':[85.7]*45,'Close':[86.0]*45,'Volume':[300]*45},
          index=pd.date_range(frame.index[-1]+pd.Timedelta(days=1), periods=45))
    future=pd.concat([frame,far_later])
    assert not detect_bottoms(future,3)['candidates'] or 'C파 저점' not in kinds(future)


def test_bad_ohlc_excluded():
    frame=candles([100,145,118,122])
    frame.loc[frame.index[-1],'Low']=float(frame.iloc[-1].High)+1
    with pytest.raises(ValueError,match='잘못된'):
        detect_bottoms(frame)


def test_time_confirm_cannot_appear_at_low_bar():
    frame=candles([160,100,132,82,86],7)
    full=kinds(frame)
    assert 'C파 저점' in full
    bottom_date=full['C파 저점']['observed_low_date']
    at_low=frame.loc[:bottom_date]
    if len(at_low)>45:
        early=kinds(at_low)
        assert not ('C파 저점' in early and early['C파 저점']['confirmed'])
    if full['C파 저점']['confirmed']:
        assert full['C파 저점']['confirmed_asof'] != bottom_date


def test_audit_idempotent_and_forward_only(tmp_path):
    frame=candles([160,100,132,82,86],7)
    t=datetime(2026,11,2,tzinfo=timezone.utc)
    result=update_bottom_audit({'EX':frame},root=tmp_path,now=t,order=3)
    assert result['new_candidates']>=1
    assert result['evaluated_5']==0
    again=update_bottom_audit({'EX':frame},root=tmp_path,now=t,order=3)
    assert again['new_candidates']==0
    # After exactly five new completed bars, prospective horizons settle.
    future_vals=np.array([88,90,89,92,93,94.],dtype=float)
    fut=pd.DataFrame({'Open':future_vals,'High':future_vals*1.004,
                      'Low':future_vals*.996,'Close':future_vals,
                      'Volume':1000},index=pd.date_range(frame.index[-1]+pd.Timedelta(days=1),periods=6))
    next_=update_bottom_audit({'EX':pd.concat([frame,fut])},root=tmp_path,now=t,order=3)
    assert next_['evaluated_5']>=1
    assert next_['evaluated_21']==0
    records=[json.loads(line) for line in (tmp_path/'reports'/'wave_bottom_forward_journal.jsonl').read_text().splitlines()]
    assert all(r['research_only'] and '5' in r['outcomes'] for r in records if r['bar_date']==str(frame.index[-1])[:10])
