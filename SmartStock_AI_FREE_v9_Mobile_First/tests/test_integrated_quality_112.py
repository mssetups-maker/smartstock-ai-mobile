"""Regression/evidence tests: no fake percentages and point-in-time labels."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'SmartStock_AI_FREE_v9_Mobile_First'))
from engine import indicators
from quality_guard import verify_and_rebuild,latest_signal,scenario_candidates,indicator_table
from historical_evidence import fixed_rule_history,model_history_from_audit,evidence_context,RULE_VERSION,outcome_band
from analysis_decision import decide


def prices(n=1500):
    idx=pd.bdate_range('2016-01-01',periods=n)
    t=np.arange(n)
    # Historical candles: drifting with cycles; all highs/lows envelop prices.
    c=80*np.exp(.00025*t+.08*np.sin(t/28.)+.015*np.sin(t/3.))
    o=c*(1+.002*np.sin(t/6.))
    hi=np.maximum(c,o)*1.007;lo=np.minimum(c,o)*.993
    return pd.DataFrame({'Open':o,'High':hi,'Low':lo,'Close':c,
                         'Volume':100000+10000*np.sin(t/11)},index=idx)


def test_indicators_extended_are_finite_and_bounded():
    raw=prices()
    df=indicators(raw)
    report=verify_and_rebuild(raw,df)
    assert report['status']=='정상'
    assert df['MFI14'].iloc[-1]>=0 and df['MFI14'].iloc[-1]<=100
    assert np.isfinite(df['OBV_slope20'].iloc[-1])
    assert 'MFI(14)' in indicator_table(latest_signal(df)).지표.tolist()


def test_corrupted_cached_rsi_is_recomputed_and_not_used_for_ranking():
    raw=prices();base=indicators(raw)
    broken=base.copy();broken.loc[broken.index[-1],'RSI14']=200
    out=verify_and_rebuild(raw,broken)
    assert out['status']=='재산정'
    assert any('RSI14' in note for note in out['repaired'])
    assert 0 <= out['df'].RSI14.iloc[-1] <= 100


def test_invalid_ohlc_blocks_before_scenario():
    raw=prices()
    raw.loc[raw.index[-1],'High']=raw.Close.iloc[-1]*.9
    with pytest.raises(ValueError):verify_and_rebuild(raw)


def test_no_volume_excludes_money_flow_from_vote():
    raw=prices();raw['Volume']=0
    report=verify_and_rebuild(raw)
    sig=latest_signal(report['df'],volume_reliable=report['volume_reliable'])
    assert all(v['indicator'] not in ('MFI(14)','OBV 거래량 추세') for v in sig['votes'])


def test_evidence_cases_do_not_overlap_and_use_matured_outcomes():
    raw=prices(1500);df=indicators(raw)
    cases=fixed_rule_history(raw,df,5,'TEST')
    assert len(cases)>100
    assert all(c['resolved_time']<=evidence_context('TEST',5,'rule:bull',RULE_VERSION,raw.index[-1])['asof'] for c in cases)
    positions={pd.Timestamp(x):i for i,x in enumerate(raw.index)}
    for left,right in zip(cases,cases[1:]):
        assert pd.Timestamp(left['resolved_time'])<=pd.Timestamp(right['prediction_time'])
        i=positions[pd.Timestamp(left['prediction_time']).tz_localize(None)]
        actual=raw.Close.iloc[i+5]/raw.Close.iloc[i]-1
        assert left['actual_outcome']==('상승' if actual>left['neutral_band'] else '하락' if actual<-left['neutral_band'] else '횡보')


def test_probability_requires_matching_completed_cases_not_score():
    raw=prices(2000);df=indicators(raw)
    sig=latest_signal(df)
    context=evidence_context('TEST',5,'rule:'+sig['bucket'],RULE_VERSION,raw.index[-1])
    candidates=scenario_candidates(sig,.99)
    allcases=fixed_rule_history(raw,df,5,'TEST')
    out=decide(candidates,context,allcases,min_samples=80)
    assert out['top'] is not None
    assert out['sample_count']==sum(x['signal_bucket']==context['signal_bucket'] for x in allcases)
    if out['probabilities']:
        assert sum(out['probabilities'].values())==pytest.approx(1)
        assert out['sample_count']>=80
        assert set(out['probabilities'])=={'상승','하락','횡보'}
    short=decide(candidates,context,allcases[:2],min_samples=80)
    assert short['probabilities'] is None
    assert short['top']['rank_score'] is not None  # rank score is NOT a percent


def test_rule_sensitivity_proof_only_prior_bars():
    raw=prices(700)
    d1=indicators(raw.iloc[:550])
    d2=indicators(raw)
    assert latest_signal(d1)['bucket']==latest_signal(d2,row=549)['bucket']
    assert outcome_band(raw,450,21)==pytest.approx(outcome_band(raw.iloc[:500],450,21))


def test_model_oos_rejects_forged_return_and_overlap():
    raw=prices(900)
    rows=[]
    for i in [500,505,525]:
        r=raw.Close.iloc[i+5]/raw.Close.iloc[i]-1
        rows.append({'날짜':str(raw.index[i].date()),'모델상승점수':.7,'실제수익률':float(r)})
    rows[1]['실제수익률']=.999  # fraudulent return, skipped
    result=model_history_from_audit(rows,raw,'TEST',5)
    assert len(result)==2
    assert all(r['model_version']=='strict-rf-v1' for r in result)


def test_volatility_threshold_uses_only_past():
    raw=prices(600)
    one=outcome_band(raw,350,5)
    raw.loc[raw.index[370:],'Close']*=4
    # The later price move must never change a past event's class band.
    assert outcome_band(raw,350,5)==pytest.approx(one)
