"""Unattended *scheduled* market observations and safeguarded research.

This is not a continuous server, market-data redistributor or trading bot.
GitHub Actions calls it; all results include collection time and errors.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from crypto_transfer import COINS, get_candles, transfer_audit
from crypto_intraday import test_one_hour
from engine import clean_prices

ROOT=Path(__file__).resolve().parent

def _dump(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def _read_watchlist():
    f=ROOT/'config'/'watchlist.txt'
    return [x.strip().upper() for x in f.read_text().splitlines() if x.strip() and not x.startswith('#')][:12]

def hourly(*,root:Path=ROOT,coin_getter=get_candles,stock_getter=None,now=None):
    now=now or datetime.now(timezone.utc)
    results=[]; failures=[]; intraday={}
    for coin in COINS:
        try:
            file=root/'data'/'crypto_5min'/f'{coin}.csv.gz'
            meta=root/'data'/'crypto_5min'/f'{coin}.backfill.json'
            old=pd.DataFrame()
            if file.exists():
                old=clean_prices(pd.read_csv(file,index_col=0,parse_dates=True))
            # Fetch a long block once, and then recent bars only.
            requested=2600 if not meta.exists() else 180
            candles=coin_getter(coin,unit='minutes/5',count=requested,now=now)
            if candles.empty: raise ValueError('완성된 캔들이 없습니다.')
            merged=clean_prices(pd.concat([old,candles])) if len(old) else clean_prices(candles)
            report=test_one_hour(merged)
            intraday[coin]=report
            # Commit at most one 5-minute training snapshot per UTC hour.
            current_hour=now.isoformat()[:13]
            should_persist=not file.exists() or not (root/'data'/'crypto_5min'/f'{coin}.last_hour.txt').exists() or (root/'data'/'crypto_5min'/f'{coin}.last_hour.txt').read_text()!=current_hour
            if should_persist:
                file.parent.mkdir(parents=True,exist_ok=True)
                merged.to_csv(file,compression='gzip',float_format='%.10g')
                (file.parent/f'{coin}.last_hour.txt').write_text(current_hour,encoding='utf-8')
                if not meta.exists():
                    _dump(meta,{'first_backfill_utc':now.isoformat(),
                                'requested':requested,'received':len(candles),
                                'first_available':merged.index[0].isoformat()})
            current=candles.tail(28)
            results.append({'market':coin,'close':round(float(current.Close.iloc[-1]),5),
               'last_candle_utc':current.index[-1].isoformat(),
               'last_1h_return_pct':round(float((current.Close.iloc[-1]/current.Close.iloc[max(0,len(current)-13)]-1)*100),3),
               'sample_count':len(current),'saved_5min_rows':len(merged),
               'intraday_research_status':report['status']})
        except Exception as e:
            failures.append({'source':coin,'error':str(e)[:180]})
    if intraday:
        _dump(root/'reports'/'crypto_intraday_latest.json',
              {'timestamp_utc':now.isoformat(),'research_only':True,
               'crypto_5min_evaluations':intraday})
    if stock_getter is None:
        import yfinance as yf
        stock_getter=lambda t: yf.Ticker(t).history(period='5d',interval='1h',auto_adjust=True,timeout=16)
    equities=[]
    for symbol in _read_watchlist()[:6]:
        try:
            raw=clean_prices(stock_getter(symbol))
            if raw.empty: raise ValueError('가격 기록이 없습니다.')
            equities.append({'symbol':symbol,'last_observed_bar':raw.index[-1].isoformat(),
                'close':round(float(raw.Close.iloc[-1]),4),'source':'무료 Yahoo: 지연 또는 불완전 가능'})
        except Exception as e:
            failures.append({'source':symbol,'error':str(e)[:180]})
    payload={'timestamp_utc':now.isoformat(),'mode':'15분 예약 감시 (실시간 아님)',
        'cryptocurrency':results,'stocks':equities,'failures':failures,
        'status':'ok' if len(results)+len(equities)>0 else 'failed',
        'last_successful_observations':len(results)+len(equities)}
    # Save new snapshot only once per UTC hour, otherwise avoid 96 Git commits/day.
    path=root/'data'/'monitor_latest.json'
    prior=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if (not prior or prior.get('timestamp_utc','')[:13]!=now.isoformat()[:13]
            or prior.get('status')=='failed'):
        _dump(path,payload)
    print(json.dumps(payload,ensure_ascii=False))
    if payload['status']=='failed':raise RuntimeError('주식과 코인 데이터를 모두 받지 못했습니다. 로그를 확인하세요.')
    return payload


def _read_cached_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return clean_prices(pd.read_csv(path, index_col=0, parse_dates=True))
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'기존 코인 가격 기록 오류: {exc}') from exc


def _append_daily_audit(path:Path, payload:dict):
    """Keep one compact immutable-looking summary per UTC date and mode.

    GitHub can rerun the daily schedule; upsert by date so retried jobs do
    not count as additional historical observations. Only audit metadata
    is saved here; data files hold the OHLCV training input separately.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    events=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                item=json.loads(line)
                if isinstance(item,dict): events.append(item)
            except ValueError:
                continue
    audit={'date_utc':payload['timestamp_utc'][:10],
           'timestamp_utc':payload['timestamp_utc'],
           'symbols_checked':len(payload['results']),
           'crypto_coverage':payload.get('crypto_coverage',{}),
           'qualified_candidates':sum(r.get('status','').startswith('연구상 성능 개선 후보') for r in payload['results']),
           'testable_experiments':sum('stock_holdout_brier' in r for r in payload['results']),
           'errors':payload.get('errors',[]),
           'results':[{k:v for k,v in r.items() if k in (
                'symbol','horizon_days','status','reason','stock_training_cases',
                'crypto_training_cases','stock_holdout_brier','stock_crypto_holdout_brier',
                'effective_holdout_cases')} for r in payload['results']]}
    events=[e for e in events if e.get('date_utc')!=audit['date_utc']]
    events.append(audit)
    # Bound file growth; one year's worth is adequate for first stage.
    path.write_text('\n'.join(json.dumps(e,ensure_ascii=False,default=str) for e in events[-366:])+'\n',encoding='utf-8')


def daily(*,root:Path=ROOT,coin_getter=get_candles,stock_getter=None,now=None):
    now=now or datetime.now(timezone.utc)
    reports=[];failure=[];crypto={};coverage={}
    for c in COINS:
        try:
            file=root/'data'/'crypto_daily'/f'{c}.csv'
            meta=root/'data'/'crypto_daily'/f'{c}.backfill.json'
            old=_read_cached_candles(file)
            # The previous 1300-day window began too late for long-history
            # stock research cutoffs. Backfill 3,800 bars ONCE, then fetch
            # only the recent bars in later runs. Exchange history may be
            # shorter than 3,800 because the pair was listed later.
            deep=not meta.exists()
            requested=3800 if deep else 10
            new=coin_getter(c,unit='days',count=requested,now=now)
            new=clean_prices(new)
            merged=clean_prices(pd.concat([old,new]).sort_index()) if len(old) else new
            if len(merged)<250:
                raise ValueError(f'일봉 자료 {len(merged)}개로 부족합니다.')
            crypto[c]=merged
            file.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(file,float_format='%.10g')
            if deep:
                _dump(meta,{'first_backfill_utc':now.isoformat(),
                    'requested':requested,'received':len(new),
                    'first_available':merged.index[0].isoformat(),
                    'last_available':merged.index[-1].isoformat()})
            coverage[c]={'rows':len(merged),'first':merged.index[0].isoformat(),
                'last':merged.index[-1].isoformat(),'deep_backfill':deep}
        except Exception as exc:
            failure.append({'source':c,'error':str(exc)[:180]})
    if stock_getter is None:
        import yfinance as yf
        stock_getter=lambda t: yf.Ticker(t).history(period='10y',interval='1d',auto_adjust=True,timeout=22)
    # Model comparisons remain in their own development + untouched holdout
    # windows. Results are research only; do NOT deploy them to live UI.
    horizons=(5,21,63)
    for symbol in _read_watchlist()[:12]:
        try:
            raw=clean_prices(stock_getter(symbol))
            if len(raw)<320:raise ValueError(f'주식 가격 자료가 {len(raw)}일뿐입니다.')
            for h in horizons:
                result=transfer_audit(raw,crypto,horizon=h)
                result['symbol']=symbol
                reports.append(result)
        except Exception as exc:
            failure.append({'source':symbol,'error':str(exc)[:180]})
    # Prioritize the 21-session forecast in the existing phone status view.
    reports.sort(key=lambda r:(r.get('horizon_days')!=21, r.get('symbol',''),r.get('horizon_days',0)))
    payload={'timestamp_utc':now.isoformat(),
             'mode':'일일 주식·코인 전이학습: 5/21/63거래일, 독립 검증',
             'status':'ok' if reports and not failure else 'partial' if reports else 'failed',
             'integrated_into_live_stock_predictions':False,
             'why':'연구 결과만 저장하며, 독립 검증과 실제 미래 성과가 확인되기 전에는 실거래 모델을 교체하지 않습니다.',
             'crypto_coverage':coverage,'horizons':list(horizons),
             'results':reports,'errors':failure}
    _dump(root/'reports'/'crypto_transfer_latest.json',payload)
    _append_daily_audit(root/'reports'/'crypto_transfer_history.jsonl',payload)
    print(json.dumps(payload,ensure_ascii=False,default=str))
    if not reports:
        raise RuntimeError('전이학습 실험을 실행하지 못했습니다. 데이터 연결을 확인하세요.')
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['hourly','daily'],required=True)
    mode=p.parse_args().mode
    (hourly if mode=='hourly' else daily)()

if __name__=='__main__':main()
