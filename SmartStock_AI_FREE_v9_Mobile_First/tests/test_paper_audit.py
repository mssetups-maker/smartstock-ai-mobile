"""10.8 prospective journal protects against hindsight signals and misleading fills."""
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from paper_audit import update_paper_audit,_settle,_closed_date,_wilson,ENGINE_VERSION

ET=ZoneInfo('America/New_York')

def candle_frame(n=260):
    dates=pd.bdate_range('2025-01-02',periods=n)
    c=100+np.arange(n)*.15+np.sin(np.arange(n)/6)*3
    d=pd.DataFrame({'Open':c-.2,'High':c+1.2,'Low':c-1.2,
                    'Close':c,'Volume':np.full(n,100000.)},index=dates)
    d.iloc[-1,d.columns.get_loc('Close')]=d.Close.iloc[-2]+5
    d.iloc[-1,d.columns.get_loc('Open')]=d.Close.iloc[-1]-1
    d.iloc[-1,d.columns.get_loc('High')]=d.Close.iloc[-1]+1
    d.iloc[-1,d.columns.get_loc('Low')]=d.Close.iloc[-1]-2
    d.iloc[-1,d.columns.get_loc('Volume')]=600000
    return d

def after(d):
    return datetime.combine(d.index[-1].date(),time(17,10),ET)

def test_utc_market_close_gate():
    t=datetime(2025,10,20,14,0,tzinfo=ET)
    assert _closed_date(t)==pd.Timestamp('2025-10-19')
    t=datetime(2025,10,20,17,0,tzinfo=ET)
    assert _closed_date(t)==pd.Timestamp('2025-10-20')


def test_no_backdated_signal_if_bar_unclosed(tmp_path):
    d=candle_frame()
    now=datetime.combine(d.index[-1].date(),time(13,45),ET)
    s=update_paper_audit({'NVDA':d},root=tmp_path,now=now,horizons=(5,))
    assert s['total_logged']==0


def test_first_run_logs_waiting_only_and_dedups(tmp_path):
    d=candle_frame()
    a=update_paper_audit({'NVDA':d},root=tmp_path,now=after(d),horizons=(5,))
    assert a['pending'] == 1 and a['settled'] == 0
    assert a['new_candidates']==1
    path=tmp_path/'reports'/'paper_strategy_journal.jsonl'
    first=path.read_text()
    b=update_paper_audit({'NVDA':d},root=tmp_path,now=after(d),horizons=(5,))
    assert b['total_logged']==1 and b['new_candidates']==0
    assert path.read_text()==first
    row=json.loads(first.splitlines()[0])
    assert row['signal_date']==str(d.index[-1].date())
    assert row['status']=='waiting_entry' and row['engine']==ENGINE_VERSION
    assert 'score' not in row and not row['snapshot_verified']


def test_next_day_gap_stop_and_fixed_snapshot(tmp_path):
    d=candle_frame();now=after(d)
    update_paper_audit({'NVDA':d},root=tmp_path,now=now,horizons=(5,))
    path=tmp_path/'reports'/'paper_strategy_journal.jsonl'
    row=json.loads(path.read_text())
    nextdate=d.index[-1]+pd.offsets.BDay(1)
    op=row['stop']*.88
    extra=pd.DataFrame({'Open':[op], 'High':[op*1.02], 'Low':[op*.98],
                       'Close':[op*.99],'Volume':[100000.]},index=[nextdate])
    later=pd.concat((d,extra)); done=update_paper_audit({'NVDA':later},root=tmp_path,
        now=datetime.combine(nextdate.date(),time(17,40),ET),horizons=(5,))
    r=[x for x in path.read_text().splitlines() if json.loads(x)['id']==row['id']][0]
    observed=json.loads(r)
    # A huge overnight break can invalidate a split-sensitive audit instead of
    # fabricating a neat fill; within-range gaps use the next open.
    assert observed['status']=='settled'
    assert observed['exit_type']=='gap_stop'
    assert observed['exit_price']==pytest.approx(op)
    assert observed['net_return_pct']<0
    assert done['settled']==1
    # Subsequent provider revisions cannot overwrite an already settled record
    changed=later.copy();changed.iloc[-1,changed.columns.get_loc('Open')]=op*.99
    update_paper_audit({'NVDA':changed},root=tmp_path,
        now=datetime.combine(nextdate.date(),time(17,40),ET),horizons=(5,))
    re=json.loads(path.read_text().splitlines()[0])
    assert re['exit_price']==observed['exit_price']


def test_same_bar_stop_first_and_horizon_exit():
    row={'id':'test','engine':ENGINE_VERSION,'symbol':'NVDA','signal_date':'2025-02-03',
         'status':'waiting_entry','signal_close':100.,'stop':95.,'target':110.,'cost_pct':.3,
         'horizon':2}
    idx=pd.to_datetime(['2025-02-03','2025-02-04','2025-02-05'])
    d=pd.DataFrame({'Open':[100,100,100],'High':[101,120,104],
                    'Low':[99,90,98],'Close':[100,102,101],'Volume':[100,100,100]},index=idx)
    result=_settle(row.copy(),d)
    assert result['status']=='settled' and result['exit_type']=='stop_first_if_both'
    assert result['exit_price']==95. and result['net_return_pct'] ==pytest.approx(-5.3)
    d.iloc[1,d.columns.get_loc('High')]=104
    d.iloc[1,d.columns.get_loc('Low')]=98
    result=_settle(row.copy(),d)
    assert result['exit_type']=='time_exit' and result['exit_price']==101.


def test_expired_horizon_stays_open_until_complete():
    row={'id':'test','engine':ENGINE_VERSION,'symbol':'NVDA','signal_date':'2025-02-03',
         'status':'waiting_entry','signal_close':100.,'stop':95.,'target':110.,'cost_pct':.3,
         'horizon':3}
    idx=pd.to_datetime(['2025-02-03','2025-02-04'])
    d=pd.DataFrame({'Open':[100,100],'High':[102,102],'Low':[98,98],
                    'Close':[100,101],'Volume':[100,100]},index=idx)
    r=_settle(row,d)
    assert r['status']=='open' and 'net_return_pct' not in r


def test_corrupt_journal_not_silently_reset(tmp_path):
    path=tmp_path/'reports'/'paper_strategy_journal.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('{garbage\n')
    with pytest.raises(ValueError,match='파손'):
        update_paper_audit({'NVDA':candle_frame()},root=tmp_path,now=after(candle_frame()))
    assert path.read_text()=='{garbage\n'


def test_bogus_data_is_excluded_and_reported(tmp_path):
    d=candle_frame();d.iloc[-1,d.columns.get_loc('High')]=-2
    out=update_paper_audit({'NVDA':d},root=tmp_path,now=after(d))
    assert out['tracked_symbols']==0 and out['errors'] and out['total_logged']==0


def test_wilson_interval_and_minimum_sample():
    assert _wilson(0,0) is None
    lo,hi=_wilson(5,10)
    assert .1<lo<.5<hi<.9


def test_daily_integration_writes_prospective_summary(tmp_path,monkeypatch):
    """Scheduled job works from a repository-specific root without network."""
    import auto_learning as job
    idx=pd.date_range('2024-01-01',periods=400,freq='D')
    values=np.linspace(100,130,len(idx))
    coin=pd.DataFrame({'Open':values-.3,'High':values+1,'Low':values-1,
                       'Close':values,'Volume':np.full(len(idx),900.)},index=idx)
    stock=candle_frame(360)
    watch=tmp_path/'config'/'watchlist.txt'
    watch.parent.mkdir(parents=True)
    watch.write_text('NVDA\n',encoding='utf-8')
    monkeypatch.setattr(job,'transfer_audit',lambda raw,crypto,horizon: {'status':'research_only','horizon_days':horizon})
    monkeypatch.setattr(job,'research_batch',lambda markets,timestamp: {
        'status':'ok','errors':[],'markets':{
            'stocks':{'experiments':[],'validated_research_candidates':0,'total_holdout_cases':0}}})
    monkeypatch.setattr(job,'save_research',lambda path,payload:None)
    monkeypatch.setattr(job,'update_forward_journal',lambda *a,**k:{})
    result=job.daily(root=tmp_path,now=after(stock),
                     coin_getter=lambda *args,**kwargs:coin,
                     stock_getter=lambda symbol:stock)
    assert result['paper_strategy_audit']['status']=='ok'
    assert result['paper_strategy_audit']['tracked_symbols']==1
    assert (tmp_path/'reports'/'paper_strategy_summary.json').exists()
    assert (tmp_path/'reports'/'crypto_transfer_latest.json').exists()
