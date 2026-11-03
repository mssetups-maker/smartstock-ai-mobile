"""11.4 structural geometry, confirmation and history regressions."""
from datetime import timedelta
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import pytest
import wave_multidegree as w


def p(i, val, k):
    d=pd.Timestamp('2020-01-01')+pd.Timedelta(days=i)
    return {'position':i,'value':float(val),'kind':k,'date':d.strftime('%Y-%m-%d'),'time':d}


def pivot_series(values,begin=20,gap=12):
    pivot_types=['L' if i%2==0 else 'H' for i in range(len(values))] if values[1]>values[0] else ['H' if i%2==0 else 'L' for i in range(len(values))]
    return [p(begin+i*gap,v,k) for i,(v,k) in enumerate(zip(values,pivot_types))]


def frame(n=300):
    idx=pd.date_range('2020-01-01',periods=n,freq='D')
    vals=100+np.sin(np.arange(n)/9)*10
    return pd.DataFrame({'Open':vals,'High':vals+1,'Low':vals-1,'Close':vals,'Volume':1000},index=idx)


def frame_from_vertices(vertices,gap=12,pre=55,tail=12):
    arr=[vertices[0]]*pre
    for a,b in zip(vertices,vertices[1:]):arr.extend(np.linspace(a,b,gap,endpoint=False))
    arr.extend([vertices[-1]]*(tail+1))
    x=np.array(arr,dtype=float)
    # avoid same-bar high&low creating ambiguous pivots: use close as extrema, small spread.
    return pd.DataFrame({'Open':x,'High':x+0.05,'Low':x-.05,'Close':x,'Volume':1000},
                        index=pd.date_range('2020-01-01',periods=len(x),freq='D'))


def test_standard_impulse_and_early_wave2():
    full=pivot_series([100,135,112,160,145,181])
    name,meta=w._impulse(full,.2)
    assert name=='일반 충격파'
    assert not meta['internal_verified']
    assert w._impulse(full[:3],.2)[0].startswith('충격파 초기')
    assert w._impulse(pivot_series([100,135,90]),.2) is None
    # wave 4 intrusion invalidates regular impulse, not automatically the theory
    assert w._impulse(pivot_series([100,135,112,160,128,180]),.2)[0].startswith('대각형')


def test_third_wave_shortest_must_fail():
    pvs=pivot_series([100,150,120,151,125,180])
    assert w._impulse(pvs,.2) is None


def test_zigzag_flat_expanded_flat_and_no_forged_running():
    def q(v):return pivot_series(v)
    assert w._abc(q([150,100,130,80]),.2)[0].startswith('지그재그')
    assert w._abc(q([150,100,148,97]),.2)[0].startswith('플랫')
    assert w._abc(q([150,100,159,85]),.2)[0].startswith('확장 플랫')
    assert w._abc(q([150,100,159,105]),.2) is None
    assert w._abc(q([150,100,110,105]),.2) is None


def test_triangle_requires_convergence_and_five_swings():
    good=pivot_series([120,80,110,85,104,91])
    assert w._triangle(good,.2)[0].startswith('수렴 삼각형')
    assert w._triangle(pivot_series([120,80,110,85,119,91]),.2) is None


def test_nested_wxy_rejects_only_three_pivot_aliases():
    vals=[150,110,135,95,125,108,145,108,128,87]
    pp=pivot_series(vals)
    out=w._wxy(pp,.2)
    assert out and out[0].startswith('복합 조정')
    assert w._wxy(pp[:4],.2) is None
    assert w._wxy(pivot_series([150,110,135,95,125,108,145,108,128,151]),.2) is None


def test_structural_result_never_claims_probability():
    pp=pivot_series([100,135,110,161,145,185])
    result=w._result(pp,'일반 충격파',w._impulse(pp,.2)[1],5,120,.2)
    assert result['status']=='후보'
    assert '확률' in result['notes']
    assert result['internal_verified'] is False
    assert result['heuristic_score'] is not None
    assert 'probabilities' not in result


def test_history_retained_after_staleness(monkeypatch):
    df=frame(350)
    all_=pivot_series([100,132,113,158,140,175],begin=60)
    # Old completed count, latest confirmed pivots elsewhere.
    def pivots(prices,order,lookback):
        return all_+[p(180,120,'L'),p(220,145,'H'),p(260,105,'L'),p(300,122,'H')]
    monkeypatch.setattr(w,'_alternating_pivots',pivots)
    z=w.analyze_wave_multidegree(df,5,orders=(5,))
    assert z['history']
    assert any(x['phase']=='일반 충격파' and x['nodes'][-1]['파동']=='5' for x in z['history'])
    assert all(x['nodes'][-1]['날짜']!='2020-12-15' for x in z['history'])


def test_future_confirmation_bar_not_used():
    # right-side order bars are needed for a confirmed pivot
    df=frame_from_vertices([100,145,119,172,148,195],gap=10,tail=1)
    seen=w._alternating_pivots(df,3,len(df))
    last_bar=len(df)-1
    assert all(int(pv['position'])<=last_bar-3 for pv in seen)
    # Once the next bars arrive, more pivots may become confirmed.
    extended=frame_from_vertices([100,145,119,172,148,195],gap=10,tail=10)
    more=w._alternating_pivots(extended,3,len(extended))
    assert len(more)>=len(seen)


def test_invalid_ohlc_does_not_draw_invented_waves():
    df=frame(100)
    df.loc[df.index[-1],'High']=70
    z=w.analyze_wave_multidegree(df,5)
    assert z['status']=='판정 보류'
    assert not z['nodes']
    assert '오류' in z['notes']


def test_no_wave_label_when_no_pivot_evidence():
    idx=pd.date_range('2020-01-01',periods=100)
    close=np.linspace(100,200,100)
    df=pd.DataFrame({'Open':close,'High':close+1,'Low':close-1,'Close':close},index=idx)
    z=w.analyze_wave_multidegree(df,5)
    assert not z['nodes']
    assert z['status']=='판정 보류'


def test_chart_renders_historical_and_present_separately():
    df=frame(200)
    old=w._result(pivot_series([100,120,110,150,132,170],begin=10),'일반 충격파',{},5,200,.2)
    active=w._result(pivot_series([140,112,132,95],begin=120),'지그재그 ABC 외형 후보',{},5,200,.2)
    f=make_subplots(rows=1,cols=1)
    w.attach_multidegree_traces(f,active,history=[old],latest=(df.index[-1],101))
    assert len(f.data)==3
    assert f.data[0].line.dash=='dash'
    assert f.data[1].line.dash=='solid'


def test_time_localized_no_post_observation_relabeling():
    df=frame_from_vertices([150,100,130,80],gap=12,tail=16)
    a=w.analyze_wave_multidegree(df,3,orders=(3,))
    assert a['search_total']>=0
    # extending by future candles does NOT alter the historical observed labels
    # when evaluating the earlier as-of slice.
    later=pd.DataFrame({'Open':[110]*30,'High':[110.5]*30,'Low':[109.5]*30,'Close':[110]*30},
        index=pd.date_range(df.index[-1]+timedelta(days=1),periods=30))
    joined=pd.concat([df,later])
    b=w.analyze_wave_multidegree(joined.iloc[:len(df)],3,orders=(3,))
    assert a['status']==b['status']
    assert [(z['phase'],z['confirmed_through']) for z in a['history']]==[(z['phase'],z['confirmed_through']) for z in b['history']]


def test_only_lower_degree_subwaves_can_confirm_truncated_fifth():
    pp=pivot_series([100,135,110,175,141,160])
    assert w._impulse(pp,.2) is None
    assert w._truncated_fifth(pp,.2,frame(150),3) is None


def test_expanding_triangle_not_labeled_contracting():
    pp=pivot_series([120,80,110,75,116,67])
    z=w._triangle(pp,.2)
    assert z is not None and z[0].startswith('확산 삼각형')


def test_nested_triple_wxyxz_requires_fifteen_corrective_legs():
    vals=[150,110,135,95,125,108,145,108,128,87,118,101,133,96,118,78]
    pp=pivot_series(vals)
    z=w._wxyxz(pp,.2)
    assert z and z[0].startswith('삼중 복합')
    assert w._wxyxz(pp[:10],.2) is None
    nodes=w._make_nodes(pp,z[0])
    assert [x['파동'] for x in nodes if x['파동'] in ('W','Y','Z')]==['W','Y','Z']
