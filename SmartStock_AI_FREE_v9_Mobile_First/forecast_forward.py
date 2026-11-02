"""Prospective ML forecast scoring, independent of trading-signal paper ledger.

A snapshot is written from a completed US daily candle only BEFORE the next
regular session's opening. The saved score is never retrained retroactively.
Actual directional outcome appears only after the selected horizon has elapsed.
No orders, no automatic model promotion, no calibration claims from tiny samples.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import numpy as np
import pandas as pd

from paper_audit import ET, _closed_date, _asof_frame, _read, _atomic_jsonl

FORECAST_VERSION='10.8-RF-prospective-v1'
HORIZONS=(5,21)


def _available_to_record(last_date,now):
    ny=now.astimezone(ET)
    last_date=pd.Timestamp(last_date).date()
    if (_closed_date(now)-pd.Timestamp(last_date)).days > 4:return False
    if ny.date()>last_date and ny.weekday()<5 and (ny.hour,ny.minute)>=(9,30):return False
    return True


def _predict_asof(frame:pd.DataFrame,horizon:int):
    """Return a research score from as-of price history, *not calibrated p*."""
    from engine import analyze, indicators
    result=analyze(indicators(frame),horizon_days=horizon)
    if result.probability is None:return None
    return {'score':float(result.probability),
            'historical_test_cases':int(result.test_cases),
            'historical_accuracy':float(result.backtest_accuracy) if result.backtest_accuracy is not None else None,
            'historical_brier':float(result.backtest_brier) if result.backtest_brier is not None else None}


def _finish(row,frame):
    if row.get('status') in ('settled','invalidated'):return row
    signal=pd.Timestamp(row['signal_date'])
    if signal not in frame.index.normalize():return row
    ahead=frame.loc[frame.index.normalize()>signal].head(int(row['horizon']))
    if len(ahead)<int(row['horizon']):return row
    observed=float(ahead.Close.iloc[-1]);ref=float(row['signal_close'])
    if not math.isfinite(observed) or ref<=0 or not .65<=observed/ref<=1.6:
        row.update(status='invalidated',reason='이상 가격 변동/분할 조정 의심');return row
    realized=observed/ref-1
    true=int(realized>0)
    score=float(row['score'])
    row.update(status='settled',evaluation_date=str(ahead.index[-1].date()),
               realized_return_pct=round(realized*100,5),actual_up=true,
               score_brier=round((score-true)**2,6),
               right_direction=int((score>=.5)==bool(true)))
    return row


def _summary(records,now):
    settled=[r for r in records if r.get('status')=='settled']
    def agg(rows):
        n=len(rows)
        return {'n':n,'accuracy_pct':round(100*np.mean([r['right_direction'] for r in rows]),2) if n else None,
                'brier':round(float(np.mean([r['score_brier'] for r in rows])),4) if n else None,
                'baseline_0_5_brier':0.25 if n else None,
                'avg_predicted_score':round(float(np.mean([r['score'] for r in rows])),4) if n else None}
    by_horizon={str(h):agg([r for r in settled if r['horizon']==h]) for h in HORIZONS}
    by_symbol={sym:agg([r for r in settled if r['symbol']==sym]) for sym in sorted({r['symbol'] for r in records})}
    return {'engine':FORECAST_VERSION,'last_checked_day_utc':now.astimezone(timezone.utc).date().isoformat(),
            'paper_only':True,'total_logged':len(records),'pending':sum(r.get('status')=='pending' for r in records),
            'invalidated':sum(r.get('status')=='invalidated' for r in records),
            'settled':len(settled),'by_horizon':by_horizon,'by_symbol':by_symbol,
            'reliability':'20건 이상 완료된 동일 기간을 우선 평가. 다른 기간/종목의 표본을 무조건 독립으로 보지 않음',
            'note':'모델 점수는 보정된 상승확률이 아닙니다. 검증 파일은 자동 모델 채택에 사용하지 않습니다.'}


def update_forecast_audit(stock_frames:dict[str,pd.DataFrame], *, root:Path,
                          now:datetime|None=None,predict_fn=None) -> dict:
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('now must include timezone')
    predict_fn=predict_fn or _predict_asof
    path=Path(root)/'reports'/'forward_ml_journal.jsonl'
    existing=_read(path)
    records={r['id']:r.copy() for r in existing}
    frames={};errors=[]
    for sym,p in stock_frames.items():
        try:
            if sym.upper().endswith(('.KS','.KQ','.T')):
                raise ValueError('미국장 기준 연구 엔진에서는 미지원')
            frames[sym.upper()]=_asof_frame(p,now)
        except Exception as exc:
            errors.append({'symbol':sym,'error':str(exc)[:150]})
    for key,r in records.items():
        if r.get('status')=='pending' and r.get('symbol') in frames:
            records[key]=_finish(r,frames[r['symbol']])
    created=0
    for sym,frame in frames.items():
        if not _available_to_record(frame.index[-1],now):continue
        for h in HORIZONS:
            day=str(frame.index[-1].date())
            key=f'{FORECAST_VERSION}|{sym}|{day}|{h}'
            if key in records:continue
            try:
                result=predict_fn(frame,h)
                if not result:continue
                score=float(result['score'])
                if not math.isfinite(score) or not 0<=score<=1:
                    raise ValueError('점수 범위 오류')
                records[key]={
                   'id':key,'engine':FORECAST_VERSION,'symbol':sym,'signal_date':day,
                   'recorded_utc':now.astimezone(timezone.utc).isoformat(),
                   'signal_close':round(float(frame.Close.iloc[-1]),7),
                   'horizon':h,'score':round(score,7),'status':'pending',
                   'prior_test_cases':int(result.get('historical_test_cases',0)),
                   'source':'Yahoo auto_adjusted US daily bars; free/delayed; model RF-v1',
                   'score_is_calibrated':False}
                created+=1
            except Exception as exc:
                errors.append({'symbol':sym,'horizon':h,'error':str(exc)[:150]})
    updated=sorted(records.values(),key=lambda x:(x['signal_date'],x['id']))
    if updated!=existing:_atomic_jsonl(path,updated)
    summary=_summary(updated,now)
    summary.update(new_snapshots=created,tracked_symbols=len(frames),errors=errors)
    out=Path(root)/'reports'/'forward_ml_summary.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    previous=json.loads(out.read_text(encoding='utf-8')) if out.exists() else {}
    if summary!=previous:
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(tmp,out)
    return summary
