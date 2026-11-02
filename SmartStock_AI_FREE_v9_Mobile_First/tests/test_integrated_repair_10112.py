from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from preflight import check
from stock_data_provider import StockDataUnavailable, fetch_stock_daily

ROOT=Path(__file__).resolve().parents[1]


def sample(n=390):
    days=pd.bdate_range('2024-01-02',periods=n)
    c=100+np.arange(n)*0.1
    return pd.DataFrame({'Open':c-.1, 'High':c+.7,'Low':c-.7,
                         'Close':c,'Volume':np.full(n,10_000)},index=days)


def broken_yahoo(symbol):
    raise PermissionError('HTTP 401 Unauthorized Invalid Crumb')


def test_no_missing_modules_and_syntax_valid():
    r=check(ROOT)
    assert r['status']=='ok',r
    assert len(r['missing'])==0
    assert r['files_checked']>=36


def test_yahoo_ok_means_backup_not_called():
    called=[]
    def backup(_):
        called.append(True)
        raise AssertionError('backup must not be called')
    f=fetch_stock_daily('NVDA',yahoo_getter=lambda _: sample(),stooq_getter=backup)
    assert not called
    assert f.attrs['stock_data_source']=='Yahoo adjusted daily'


def test_yahoo_401_falls_back_without_fake_data():
    f=fetch_stock_daily('SPY',yahoo_getter=broken_yahoo,stooq_getter=lambda _:sample())
    assert len(f)==390
    assert f.attrs['stock_data_source']=='Stooq EOD fallback'


def test_two_data_source_errors_raise_not_fabricate():
    with pytest.raises(StockDataUnavailable,match='Yahoo.*Stooq'):
        fetch_stock_daily('NVDA',yahoo_getter=broken_yahoo,
                          stooq_getter=lambda _: pd.DataFrame())


def test_missing_one_module_prints_entire_missing_set_before_network(tmp_path):
    # True CLI behavior: the first import of a missing analytical module
    # never occurs before preflight, which reports the complete dependency set.
    (tmp_path/'auto_learning.py').write_text((ROOT/'auto_learning.py').read_text(),encoding='utf-8')
    (tmp_path/'preflight.py').write_text((ROOT/'preflight.py').read_text(),encoding='utf-8')
    proc=subprocess.run([sys.executable,'auto_learning.py','--mode','daily'],cwd=tmp_path,
                         capture_output=True,text=True,timeout=12)
    assert proc.returncode != 0
    assert 'opportunity_diagnostics.py' in proc.stdout
    assert 'pattern_discovery.py' in proc.stdout
    assert 'stock_data_provider.py' in proc.stdout
    assert 'HTTP 401' not in proc.stdout


def test_real_python_import_chain_is_loadable():
    # Imports must work without API connections or running a live job.
    import auto_learning
    import scheduled_scan
    import opportunity_diagnostics
    import wave_bottom_audit
    import pattern_discovery
    assert callable(auto_learning.daily)
    assert callable(scheduled_scan.run_scan)
    assert callable(opportunity_diagnostics.batch_diagnostics)
    assert callable(wave_bottom_audit.update_bottom_audit)
    assert callable(pattern_discovery.research_batch)


def test_stooq_csv_response_parsed_without_network(monkeypatch):
    from stock_data_provider import _stooq_csv
    class FakeResponse:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,_):
            return (b'Date,Open,High,Low,Close,Volume\n'
                    b'2026-10-29,100,102,99,101,10000\n'
                    b'2026-10-30,101,103,100,102,11000\n')
    calls=[]
    def fake_urlopen(req,timeout=12):
        calls.append(req.full_url)
        return FakeResponse()
    monkeypatch.setattr('stock_data_provider.urlopen',fake_urlopen)
    out=_stooq_csv('NVDA',now=datetime(2026,11,2,tzinfo=timezone.utc))
    assert len(out)==2 and 'nvda.us' in calls[0]
    assert out.Close.iloc[-1]==102


def test_stooq_rejects_quota_response(monkeypatch):
    from stock_data_provider import _stooq_csv
    class Quota:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,_):return b'Too many requests (daily limit)'
    monkeypatch.setattr('stock_data_provider.urlopen',lambda *a,**k: Quota())
    with pytest.raises(ValueError,match='CSV 데이터를 제공하지'):
        _stooq_csv('SPY',now=datetime(2026,11,2,tzinfo=timezone.utc))
