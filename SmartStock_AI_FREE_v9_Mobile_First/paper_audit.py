"""10.8 prospective, *paper-only* strategy audit.

Records a decision made from a completed daily candle, then waits for new
candles to arrive before scoring it. No strategy is optimized or promoted from
this journal. This is NOT a broker fill or a calibrated probability.

For the US ticker watchlist only; Yahoo historical OHLCV is not guaranteed
point-in-time and can be revised, split-adjusted, or delayed. Keep audit data
separate from analysis features and never overwrite an existing decision.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from engine import clean_prices
from market_decision import evaluate_market

ET = ZoneInfo('America/New_York')
ENGINE_VERSION = '10.8-strategy-rules-v1'
DEFAULT_HORIZONS = (5, 21)
COST_PCT = 0.30


def _closed_date(now: datetime) -> pd.Timestamp:
    """Latest date that can have a *fully completed* NYSE daily candle.

    Weekends/market holidays are handled by observed bars, not a hard-coded
    holiday calendar. Provider data itself must still be checked for age.
    """
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    ny = now.astimezone(ET)
    last = ny.date() if (ny.hour, ny.minute) >= (16, 30) else ny.date()-timedelta(days=1)
    return pd.Timestamp(last)


def _valid_ohlcv(frame: pd.DataFrame) -> bool:
    if frame.empty or len(frame) < 90:
        return False
    cells = frame[['Open', 'High', 'Low', 'Close']].to_numpy(dtype=float)
    if not np.isfinite(cells).all() or (cells <= 0).any():
        return False
    if (frame.High < frame[['Open','Close','Low']].max(axis=1)).any():
        return False
    if (frame.Low > frame[['Open','Close','High']].min(axis=1)).any():
        return False
    return True


def _asof_frame(prices: pd.DataFrame, now: datetime) -> pd.DataFrame:
    data = clean_prices(prices)
    data = data.loc[data.index.normalize() <= _closed_date(now)].copy()
    if data.index.has_duplicates:
        raise ValueError('duplicate dates')
    if not _valid_ohlcv(data):
        raise ValueError('가격 데이터 결측 또는 OHLC 관계 오류; 모의매매 분석 제외')
    return data


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows=[]
    for i,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try:
            r=json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f'실전 추적 기록 {i}행 파손; 원본 보존 후 검토 필요: {e}') from e
        if not isinstance(r,dict) or not r.get('id'):
            raise ValueError(f'실전 추적 기록 {i}행 형식 오류')
        rows.append(r)
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('중복 ID; 기록을 임의로 병합하지 않습니다')
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, name=tempfile.mkstemp(prefix='.paper_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as out:
            for r in rows:
                out.write(json.dumps(r,ensure_ascii=False,allow_nan=False,sort_keys=True)+'\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.remove(name)


def _snapshot(symbol:str,h:int,frame:pd.DataFrame,now:datetime) -> dict[str,Any]|None:
    plan=evaluate_market(frame,horizon=h,score=None,quality='정상',
        strict_recommendation='판단 보류',symbol=symbol,profile='균형형')
    if not plan.candidate or not plan.strategy_key or plan.entry is None:
        return None
    day=str(frame.index[-1].date())
    # Avoid backdating stale picks when Yahoo has not delivered new bars.
    if (_closed_date(now)-frame.index[-1].normalize()).days>4:
        return None
    # If the next session has already opened, even yesterday's completed
    # bar is too late to claim a forward trade at that session's open.
    ny=now.astimezone(ET)
    if (ny.date()>frame.index[-1].date() and ny.weekday()<5
            and (ny.hour,ny.minute)>=(9,30)):
        return None
    if not (0<plan.stop<plan.entry<plan.target1):
        return None
    return {
      'id':f'{ENGINE_VERSION}|{symbol}|{day}|{h}|{plan.strategy_key}',
      'engine':ENGINE_VERSION, 'symbol':symbol, 'signal_date':day,
      'recorded_utc':now.astimezone(timezone.utc).isoformat(),
      'horizon':h,'profile':'균형형','strategy':plan.strategy_key,
      'strategy_name':plan.strategy_name,'status':'waiting_entry',
      'signal_close':round(float(plan.entry),7),'stop':round(float(plan.stop),7),
      'target':round(float(plan.target1),7),'cost_pct':COST_PCT,
      'source':'Yahoo Finance delayed daily OHLCV; rules-only, not model probability',
      'snapshot_verified':False,
      'reason':'확정 종가에서 발견한 규칙 기반 전략 후보. 다음 완성 봉 시가부터 모의 평가.',
    }


def _settle(row:dict[str,Any], df:pd.DataFrame) -> dict[str,Any]:
    """Update outcome using only bars that happened AFTER saved signal_date.

    Stop is evaluated before target on a bar when path is unknown. Gap fills
    use observed open rather than assuming execution at a better stop/target.
    """
    if row['status'] in ('settled','invalidated'):
        return row  # immutable after first completed assessment
    day=pd.Timestamp(row['signal_date'])
    # Disallow trying to reconstruct entry from a revised/missing signal bar.
    if day not in df.index.normalize():
        return row
    following=df.loc[df.index.normalize()>day].head(int(row['horizon']))
    if following.empty:
        return row
    if row.get('filled_entry') is None:
        first=following.iloc[0]
        op=float(first.Open)
        prev=float(row['signal_close'])
        # Possible split / bad backadjustment: do not compare incompatible prices.
        if not (0.65<=op/prev<=1.6):
            row.update(status='invalidated', reason='가격 단절/분할 의심: 조정된 OHLC 재검증 필요')
            return row
        row['filled_entry']=round(op,7)
        row['entry_date']=str(following.index[0].date())
        row['status']='open'
    entry=float(row['filled_entry']);stop=float(row['stop']);target=float(row['target'])
    if entry<=0 or stop<=0 or target<=stop:
        row.update(status='invalidated',reason='입력 가격 범위 오류');return row
    exit_price=None;exit_reason=None;exit_day=None
    for dt,bar in following.iterrows():
        o,lo,hi=map(float,(bar.Open,bar.Low,bar.High))
        if not (min(o,float(bar.Close))>=lo and max(o,float(bar.Close))<=hi):
            row.update(status='invalidated',reason='체결 평가 구간 OHLC 오류');return row
        if o<=stop:exit_price=o;exit_reason='gap_stop'
        elif o>=target:exit_price=o;exit_reason='gap_target'
        elif lo<=stop:exit_price=stop;exit_reason='stop_first_if_both'
        elif hi>=target:exit_price=target;exit_reason='target'
        if exit_price is not None:
            exit_day=str(dt.date());break
    if exit_price is None and len(following)>=int(row['horizon']):
        exit_price=float(following.iloc[-1].Close)
        exit_day=str(following.index[-1].date());exit_reason='time_exit'
    if exit_price is not None:
        row.update(status='settled',exit_date=exit_day,
                   exit_price=round(float(exit_price),7),exit_type=exit_reason,
                   gross_return_pct=round(100*(exit_price/entry-1),5),
                   net_return_pct=round(100*(exit_price/entry-1)-float(row['cost_pct']),5))
    return row


def _wilson(x:int,n:int,z:float=1.96) -> tuple[float,float]|None:
    if n<=0:return None
    ph=x/n;den=1+z*z/n
    mid=(ph+z*z/(2*n))/den
    err=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return (max(0,mid-err),min(1,mid+err))


def _summarize(rows:list[dict[str,Any]], *, now:datetime) -> dict[str,Any]:
    settled=[r for r in rows if r.get('status')=='settled']
    pending=[r for r in rows if r.get('status') in ('open','waiting_entry')]
    net=[float(r['net_return_pct']) for r in settled]
    wins=sum(x>0 for x in net); n=len(net)
    # The observations are overlapping watchlist ideas, not a compounded portfolio.
    conf=_wilson(wins,n)
    by_strategy={}
    for strategy in ('breakout','pullback','rebound'):
        subset=[r for r in settled if r.get('strategy')==strategy]
        vals=[float(r['net_return_pct']) for r in subset]
        by_strategy[strategy]={
            'settled':len(vals),'positive':sum(x>0 for x in vals),
            'average_net_pct':round(float(np.mean(vals)),3) if vals else None}
    by_symbol={}
    for sym in sorted({r.get('symbol','') for r in rows if r.get('symbol')}):
        vals=[float(r['net_return_pct']) for r in settled if r.get('symbol')==sym]
        by_symbol[sym]={
            'settled':len(vals),
            'pending':sum(r.get('symbol')==sym and r.get('status') in ('open','waiting_entry') for r in rows),
            'average_net_pct':round(float(np.mean(vals)),3) if vals else None,
            'positive_rate_pct':round(sum(v>0 for v in vals)/len(vals)*100,1) if vals else None}
    # A soft diagnostic, not an auto-trading kill switch: use only observable
    # realized forward outcomes, chronologically split and adequately sampled.
    drift_warning=None
    chronological=sorted(settled,key=lambda r:(r.get('exit_date',''),r['id']))
    if len(chronological)>=30:
        halfway=len(chronological)//2
        early=np.mean([float(r['net_return_pct']) for r in chronological[:halfway]])
        recent=np.mean([float(r['net_return_pct']) for r in chronological[halfway:]])
        if recent<0 and early-recent>=2.0:
            drift_warning='최근 절반 모의 성적이 앞선 절반보다 2%p 이상 낮고 음수입니다. 전략 사용 적합성 재검증 필요.'
    return {'generated_utc':now.astimezone(timezone.utc).isoformat(),
            'engine':ENGINE_VERSION,'paper_only':True,
            'total_logged':len(rows),'pending':len(pending),'settled':n,
            'invalidated':sum(r.get('status')=='invalidated' for r in rows),
            'win_rate_pct':round(wins/n*100,2) if n else None,
            'win_rate_wilson95_pct':[round(c*100,1) for c in conf] if conf else None,
            'average_net_return_pct':round(float(np.mean(net)),3) if net else None,
            'average_net_after_0_6pct_cost':round(float(np.mean(net))-(.6-COST_PCT),3) if net else None,
            'by_strategy':by_strategy,'by_symbol':by_symbol,
            'drift_warning':drift_warning,
            'assessment':'최소 20건 미만: 성적 판단 유보' if n<20 else '과거가 아닌 이후 수집된 신호의 모의 평가; 실제 수익 보장 불가',
            'limitations':'모의 단일 거래 사례의 단순 평균. 포트폴리오 수익률 아님; 시세 지연/수정/배당/분할/체결오차 주의'}


def update_paper_audit(stock_frames:dict[str,pd.DataFrame], *,root:Path,
                       now:datetime|None=None,horizons=DEFAULT_HORIZONS) -> dict[str,Any]:
    """Reconcile settled paper trades and log TODAY's new rules-only candidates.

    Save files to reports/ so existing corrected workflow git-add picks them up.
    Not called on Streamlit page views; daily GitHub job is the writer.
    """
    now=now or datetime.now(timezone.utc)
    root=Path(root)
    path=root/'reports'/'paper_strategy_journal.jsonl'
    original=_read(path)
    records={r['id']:r.copy() for r in original}  # never mutate snapshot used for change detection
    frames={};errors=[]
    for sym,p in stock_frames.items():
        try:
            if sym.upper().endswith(('.KS','.KQ','.T')):
                raise ValueError('미국장 종료 시각 기준 설계: 비미국 종목은 아직 지원하지 않음')
            frames[sym.upper()]=_asof_frame(p,now)
        except Exception as e:
            errors.append({'symbol':sym,'error':str(e)[:150]})
    # reconcile each existing observation first, even if a symbol is no longer a watchlist item
    for key,row in records.items():
        if row.get('status') not in ('settled','invalidated') and row.get('symbol') in frames:
            records[key]=_settle(row,frames[row['symbol']])
    new=0
    for sym,frame in frames.items():
        for h in horizons:
            try:
                if any(r.get('symbol')==sym and r.get('horizon')==h and
                       r.get('status') in ('open','waiting_entry') for r in records.values()):
                    continue
                candidate=_snapshot(sym,h,frame,now)
                if candidate and candidate['id'] not in records:
                    records[candidate['id']]=candidate;new+=1
            except Exception as e:
                errors.append({'symbol':sym,'horizon':int(h),'error':str(e)[:150]})
    all_rows=sorted(records.values(),key=lambda r:(r['signal_date'],r['id']))
    # Preserve the dataset if nothing changed. First run creates empty summary.
    if all_rows != original:
        _atomic_jsonl(path,all_rows)
    summary=_summarize(all_rows,now=now)
    summary.update(new_candidates=new,tracked_symbols=len(frames),errors=errors,
                   last_checked_day_utc=now.astimezone(timezone.utc).date().isoformat())
    out=root/'reports'/'paper_strategy_summary.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    # Exclude volatile generated_utc from version control on repeated same-day runs.
    previous=json.loads(out.read_text(encoding='utf-8')) if out.exists() else {}
    comparison={k:v for k,v in summary.items() if k!='generated_utc'}
    if not previous or comparison!={k:v for k,v in previous.items() if k!='generated_utc'}:
        tmp=out.with_suffix('.tmp')
        tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(tmp,out)
    return summary
