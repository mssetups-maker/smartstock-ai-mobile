"""10.8 ML prospective predictions have immutable score and delayed labels."""
import json
from datetime import datetime,time
from pathlib import Path
from zoneinfo import ZoneInfo
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from forecast_forward import update_forecast_audit, _finish, _available_to_record, FORECAST_VERSION
from test_paper_audit import candle_frame,after

ET=ZoneInfo('America/New_York')


def model_stub(prices,h):
    return {'score':.66 if h==5 else .42,'historical_test_cases':20}


def test_ml_prospective_snapshot_and_idempotence(tmp_path):
    d=candle_frame(360)
    a=update_forecast_audit({'NVDA':d},root=tmp_path,now=after(d),predict_fn=model_stub)
    assert a['pending']==2 and a['settled']==0 and a['new_snapshots']==2
    p=tmp_path/'reports'/'forward_ml_journal.jsonl'
    before=p.read_text()
    b=update_forecast_audit({'NVDA':d},root=tmp_path,now=after(d),predict_fn=model_stub)
    assert p.read_text()==before
    assert b['new_snapshots']==0 and b['total_logged']==2
    rows=[json.loads(x) for x in before.splitlines()]
    assert all(r['engine']==FORECAST_VERSION and r['status']=='pending' for r in rows)


def test_ml_outcome_waits_exact_horizon_and_preserves_frozen_score(tmp_path):
    d=candle_frame(360); update_forecast_audit({'NVDA':d},root=tmp_path,now=after(d),predict_fn=model_stub)
    path=tmp_path/'reports'/'forward_ml_journal.jsonl'
    first=[json.loads(x) for x in path.read_text().splitlines()]
    nextdays=pd.bdate_range(d.index[-1]+pd.offsets.BDay(1),periods=5)
    px=float(d.Close.iloc[-1])
    ext=pd.DataFrame({'Open':[px+1]*5,'High':[px+2]*5,'Low':[px]*5,
                      'Close':[px+1]*5,'Volume':[100000]*5},index=nextdays)
    later=pd.concat([d,ext])
    # Avoid a new same-day snapshot in test by running after next market open
    res=update_forecast_audit({'NVDA':later},root=tmp_path,
        now=datetime.combine(nextdays[-1].date(),time(17,10),ET),predict_fn=model_stub)
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    for r in rows:
        if r['signal_date']==str(d.index[-1].date()) and r['horizon']==5:
            assert r['status']=='settled' and r['right_direction']==1
            assert r['score']==.66
            assert r['score_brier']==pytest.approx((.66-1)**2)
        elif r['signal_date']==str(d.index[-1].date()) and r['horizon']==21:
            assert r['status']=='pending' and r['score']==.42
    assert res['settled']>=1
    # Revised price history must NOT recalculate a settled model prediction.
    later.iloc[-1,later.columns.get_loc('Close')]=px+2
    res2=update_forecast_audit({'NVDA':later},root=tmp_path,
        now=datetime.combine(nextdays[-1].date(),time(17,10),ET),predict_fn=model_stub)
    ree=[json.loads(x) for x in path.read_text().splitlines()]
    assert ree[0]['score']==first[0]['score']
    assert any(r['status']=='settled' and r['score_brier']==pytest.approx((.66-1)**2) for r in ree)


def test_no_retroactive_ml_signal_after_next_open(tmp_path):
    d=candle_frame(360)
    now=datetime.combine(d.index[-1].date(),time(15,0),ET)
    s=update_forecast_audit({'NVDA':d},root=tmp_path,now=now,predict_fn=model_stub)
    assert s['total_logged']==0
    assert not _available_to_record(d.index[-2],datetime.combine(d.index[-1].date(),time(15,0),ET))


def test_score_range_is_checked(tmp_path):
    d=candle_frame(360)
    out=update_forecast_audit({'NVDA':d},root=tmp_path,now=after(d),
                            predict_fn=lambda p,h:{'score':1.2})
    assert out['errors'] and out['total_logged']==0


def test_real_model_can_withhold_prediction(tmp_path):
    d=candle_frame(360)
    s=update_forecast_audit({'NVDA':d},root=tmp_path,now=after(d),
                            predict_fn=lambda p,h:None)
    assert s['total_logged']==0 and not s['errors']
