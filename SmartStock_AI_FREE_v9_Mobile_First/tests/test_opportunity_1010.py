"""10.10: paper-only opportunity funnel and fail-fast deployment protection."""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from opportunity_diagnostics import diagnose_one,batch_diagnostics,save_diagnostics,_rule_checks
from preflight import check
from market_decision import _signals,_features

ET=ZoneInfo('America/New_York')

def prices(n=320):
    i=pd.bdate_range('2025-01-02',periods=n)
    close=90+np.arange(n)*.13+4*np.sin(np.arange(n)/11)
    op=close-.4
    return pd.DataFrame({'Open':op,'High':close+2,'Low':op-2,'Close':close,
                         'Volume':np.full(n,120000.)},index=i)

def after(x):
    return datetime.combine(x.index[-1].date(),datetime.min.time(),ET).replace(hour=17,minute=20)

def test_feature_checks_match_production_on_multiple_dates():
    p=prices(400)
    data=_signals(_features(p))
    for _,row in data.iloc[80::3].iterrows():
        checks=_rule_checks(row)
        assert all(bool(row[k])==all(v.values()) for k,v in checks.items())

def test_diagnosis_contains_actual_reasons_and_no_order_price():
    p=prices()
    out=diagnose_one(p,'NVDA',now=after(p))
    assert out['bar_date']==str(p.index[-1].date())
    assert out['paper_only']
    assert set(out['strategies'])=={'breakout','pullback','rebound'}
    for strat in out['strategies'].values():
        assert strat['passed']+len(strat['missing'])==strat['total']
        assert 0<=strat['signal_days_in_lookback']<=252
        assert not strat['historical_count_is_trades']
    assert out['levels']['prior_20_day_high']>0
    assert 'entry' not in out['levels']

def test_unclosed_bar_is_not_used():
    p=prices()
    today=p.index[-1].date()
    morning=datetime.combine(today,datetime.min.time(),ET).replace(hour=12)
    out=diagnose_one(p,'NVDA',now=morning)
    assert out['bar_date']==str(p.index[-2].date())

def test_future_observations_do_not_change_past_diagnosis():
    p=prices(350)
    ref=p.iloc[:310]
    now=after(ref)
    prior=diagnose_one(ref,'NVDA',now=now)
    later=p.copy()
    # Alter unknown future while respecting OHLC relationship.
    for col in ['Open','High','Low','Close']:
        later.loc[later.index>ref.index[-1],col]*=1.15
    after_report=diagnose_one(later,'NVDA',now=now)
    assert after_report==prior

def test_malformed_candles_fail_closed():
    p=prices();p.loc[p.index[-1],'Close']=p.High.iloc[-1]+100
    with pytest.raises(ValueError,match='OHLC'):
        diagnose_one(p,'NVDA',now=after(p))

def test_duplicate_daily_save_does_not_rewrite(tmp_path):
    p=prices()
    out=batch_diagnostics({'NVDA':p},now=after(p))
    f=tmp_path/'reports'/'opportunity_funnel_latest.json'
    assert save_diagnostics(f,out)
    assert not save_diagnostics(f,out)
    later={**out,'timestamp_utc':(after(p)+timedelta(minutes=2)).isoformat()}
    assert not save_diagnostics(f,later)
    assert json.loads(f.read_text())['covered_symbols']==1

def test_partial_failures_distinct_from_success():
    p=prices();bad=p.copy();bad.loc[bad.index[-1],'High']=0
    out=batch_diagnostics({'NVDA':p,'BAD':bad},now=after(p))
    assert out['status']=='partial' and out['covered_symbols']==1
    assert out['errors'][0]['symbol']=='BAD'

def test_preflight_detects_missing_file_and_syntax_error(tmp_path):
    a=check(tmp_path)
    assert 'pattern_discovery.py' in a['missing']
    assert a['status']=='failed'
    (tmp_path/'engine.py').write_text('def broken(\n',encoding='utf-8')
    b=check(tmp_path)
    assert any('engine.py' in s for s in b['syntax_errors'])

def test_preflight_is_good_in_bundle():
    root=Path(__file__).resolve().parents[1]
    assert check(root)['status']=='ok'


def test_daily_job_writes_opportunity_report(tmp_path,monkeypatch):
    import auto_learning as job
    idx=pd.date_range('2024-01-01',periods=440,freq='D')
    vals=np.linspace(100,120,len(idx))
    coin=pd.DataFrame({'Open':vals-.2,'High':vals+2,'Low':vals-2,'Close':vals,'Volume':10000},index=idx)
    stock=prices()
    watch=tmp_path/'config'/'watchlist.txt'
    watch.parent.mkdir(parents=True)
    watch.write_text('NVDA\n')
    monkeypatch.setattr(job,'transfer_audit',lambda *a,**k: {'status':'research_only','horizon_days':k['horizon']})
    monkeypatch.setattr(job,'research_batch',lambda markets,timestamp:{'status':'ok','errors':[],'markets':{
      'stocks':{'experiments':[],'validated_research_candidates':0,'total_holdout_cases':0}}})
    monkeypatch.setattr(job,'save_research',lambda path,payload:None)
    monkeypatch.setattr(job,'update_forward_journal',lambda *a,**k:{})
    out=job.daily(root=tmp_path,now=after(stock),stock_getter=lambda symbol:stock,
                  coin_getter=lambda *a,**k:coin)
    assert out['opportunity_funnel']['status']=='ok'
    saved=json.loads((tmp_path/'reports'/'opportunity_funnel_latest.json').read_text())
    assert saved['covered_symbols']==1
    assert 'NVDA' in saved['by_symbol']

def test_chat_answers_failed_gates_from_real_snapshot():
    from explain_chat import answer_question
    p=prices()
    details=diagnose_one(p,'NVDA',now=after(p))
    answer=answer_question('왜 매수 신호 0건이야? 어떤 조건이 부족해?',
                           {'opportunity_diagnostics':details})
    assert '완료 일봉' in answer and '상방 돌파' in answer and '미충족' in answer
    assert '매수 승인' in answer


def test_forecast_summary_compares_always_up_and_overlap():
    from forecast_forward import _summary
    rows=[]
    for i in range(25):
        day=str((pd.Timestamp('2025-01-02')+pd.offsets.BDay(i)).date())
        y=int(i%3!=0)
        rows.append({'id':f'x{i}','symbol':'NVDA','signal_date':day,
                     'status':'settled','horizon':21,'score':.6,
                     'actual_up':y,'right_direction':y,'score_brier':(.6-y)**2})
    report=_summary(rows,datetime(2025,5,1,tzinfo=ZoneInfo('UTC')))
    r=report['by_horizon']['21']
    assert r['n']==25 and 0<r['nonoverlap_proxy']<25
    assert r['always_up_accuracy_pct']==r['observed_up_rate_pct']
    assert r['brier_improvement_over_fixed_half'] is not None


def test_us_daily_intraday_bar_is_excluded_before_transfer_learning(tmp_path,monkeypatch):
    from auto_learning import daily
    import auto_learning as job
    p=prices(400)
    idx=pd.date_range('2024-01-01',periods=430,freq='D')
    x=np.linspace(50,80,len(idx))
    coin=pd.DataFrame({'Open':x-.5,'High':x+1,'Low':x-1,
                       'Close':x,'Volume':100},index=idx)
    (tmp_path/'config').mkdir()
    (tmp_path/'config'/'watchlist.txt').write_text('NVDA\n')
    seen=[]
    def transfer(frame,crypto,horizon):
        seen.append(frame.index[-1])
        return {'status':'research_only','horizon_days':horizon}
    monkeypatch.setattr(job,'transfer_audit',transfer)
    monkeypatch.setattr(job,'research_batch',lambda markets,timestamp:{'status':'ok','errors':[],'markets':{'stocks':{
      'experiments':[],'validated_research_candidates':0,'total_holdout_cases':0}}})
    monkeypatch.setattr(job,'save_research',lambda *a,**k:None)
    monkeypatch.setattr(job,'update_forward_journal',lambda *a,**k:{})
    before_close=datetime.combine(p.index[-1].date(),datetime.min.time(),ET).replace(hour=12)
    result=daily(root=tmp_path,now=before_close,stock_getter=lambda sym:p,coin_getter=lambda *a,**k:coin)
    assert len(seen)==3 and all(x.date()<p.index[-1].date() for x in seen)
    assert result['opportunity_funnel']['status']=='ok'
