import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import clean_prices, indicators
from research import adaptive_signals,wave_candidates,candidate_signals,free_explanation
from journal import update_journal
from scheduled_scan import run_scan

DATA=Path(__file__).resolve().parents[1]/'demo_synthetic.csv'
def frame():return clean_prices(pd.read_csv(DATA,parse_dates=['Date'],index_col='Date'))

def test_adaptive_weights_and_score():
    d=indicators(frame())
    a=adaptive_signals(d)
    assert a.score is not None and -100<=a.score<=100
    assert len(a.rows)==8
    assert a.sample_count>=12

def test_causal_signals_no_future_data():
    d=frame()
    all_s=candidate_signals(indicators(d))
    first=candidate_signals(indicators(d.iloc[:-40]))
    np.testing.assert_allclose(all_s.loc[first.index].to_numpy(),first.to_numpy(),equal_nan=True)

def test_wave_candidate_not_claiming_certain_elliott():
    out=wave_candidates(indicators(frame()))
    assert out['label']
    assert '확정하지 않습니다' in out['notes']
    assert len(out['fib'])==7

def test_journal_only_scores_after_21_bars(tmp_path):
    p=frame()
    path=tmp_path/'predictions.csv'
    first=update_journal(path,'ABC',p.iloc[:-60],.7)
    assert len(first)==1 and pd.isna(first.loc[0,'direction_correct'])
    second=update_journal(path,'ABC',p.iloc[:-30],.7)
    assert len(second)==2 and float(second.loc[0,'direction_correct']) in (0.,1.)
    assert pd.isna(second.loc[1,'direction_correct'])
    third=update_journal(path,'ABC',p,.7)
    assert len(third)==3
    assert third['direction_correct'].notna().sum()==2

def test_scheduled_scan_with_injected_fake_prices(tmp_path):
    src=frame()
    out=run_scan(['DEMO'],root=tmp_path,fetcher=lambda t:src)
    assert len(out)==1 and out[0]['symbol']=='DEMO'
    assert (tmp_path/'data'/'predictions.csv').exists()
    assert (tmp_path/'reports'/'latest.md').exists()
