from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from preflight import check
from stock_data_provider import fetch_stock_daily, ensure_recent_daily, StockDataUnavailable
from system_check import inspect

ROOT=Path(__file__).resolve().parents[1]

def price_frame(date='2026-09-01', count=30):
    dates=pd.date_range(date,periods=count,freq='B')
    c=np.linspace(100,105,count)
    return pd.DataFrame({'Open':c,'High':c+1,'Low':c-1,'Close':c+.1,'Volume':1000},index=dates)

def test_preflight_full_module_coverage():
    status=check(ROOT)
    assert status['status']=='ok',status
    assert status['files_checked']>=37

def test_stooq_fallback_for_closed_ohlc_corruption():
    now=datetime(2026,11,3,22,tzinfo=timezone.utc)
    broken=price_frame()
    broken.loc[broken.index[-1],'High']=broken.iloc[-1].Low-5
    flag=[]
    def backup(_):
        flag.append('called')
        return price_frame()
    out=fetch_stock_daily('NVDA',now=now,yahoo_getter=lambda _:broken,stooq_getter=backup)
    assert flag
    assert out.attrs['stock_data_source']=='Stooq EOD fallback'

def test_dont_reject_unfinished_daily_bar_when_closed_history_valid():
    now=datetime(2026,11,3,18,tzinfo=timezone.utc)  # 13:00 NY, market not closed
    frame=price_frame('2026-09-01',count=40)
    # Inject an incomplete same-day bar with inconsistent transient OHLC.
    frame.loc[pd.Timestamp('2026-11-03')]=[105,94,95,106,1000]
    def fallback(_):raise AssertionError('fallback should not run')
    out=fetch_stock_daily('NVDA',now=now,yahoo_getter=lambda _:frame,stooq_getter=fallback)
    assert out.attrs['stock_data_source']=='Yahoo adjusted daily'

def test_vendor_failure_does_not_fake_prices():
    with pytest.raises(StockDataUnavailable):
        fetch_stock_daily('NVDA',yahoo_getter=lambda _:pd.DataFrame(),stooq_getter=lambda _:pd.DataFrame())

def test_recency_check_fails_closed_for_stale_history():
    with pytest.raises(ValueError,match='오래되어'):
        ensure_recent_daily(price_frame('2025-01-01'), 'NVDA',now=datetime(2026,11,3,tzinfo=timezone.utc))

def test_system_check_recognizes_stale_report(tmp_path):
    from shutil import copytree
    copytree(ROOT,tmp_path/'app',ignore=__import__('shutil').ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
    root=tmp_path/'app'
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports'/'system_health_latest.json').write_text(json.dumps({'version':'10.9','status':'ok','failure_count':0,
      'timestamp_utc':'2026-10-01T00:00:00+00:00'}),encoding='utf8')
    result=inspect(root,now=datetime(2026,11,3,tzinfo=timezone.utc))
    assert result['status']=='partial'
    assert len(result['errors'])>=2
