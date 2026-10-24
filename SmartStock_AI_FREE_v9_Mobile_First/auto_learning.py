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
    results=[]; failures=[]
    for coin in COINS:
        try:
            candles=coin_getter(coin,unit='minutes/5',count=28,now=now)
            if candles.empty: raise ValueError('완성된 캔들이 없습니다.')
            results.append({'market':coin,'close':round(float(candles.Close.iloc[-1]),5),
               'last_candle_utc':candles.index[-1].isoformat(),
               'last_1h_return_pct':round((float(candles.Close.iloc[-1]/candles.Close.iloc[max(0,len(candles)-13)]-1)*100),3),
               'sample_count':len(candles)})
        except Exception as e:
            failures.append({'source':coin,'error':str(e)[:180]})
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


def daily(*,root:Path=ROOT,coin_getter=get_candles,stock_getter=None,now=None):
    now=now or datetime.now(timezone.utc)
    reports=[];failure=[];crypto={}
    for c in COINS:
        try:
            df=coin_getter(c,unit='days',count=1300,now=now)
            if len(df)<250:raise ValueError(f'코인 일봉이 {len(df)}개뿐입니다.')
            crypto[c]=df
            f=root/'data'/'crypto_daily'/f'{c}.csv'
            f.parent.mkdir(parents=True,exist_ok=True)
            # In-place logical dedup. The Git text delta remains small each day.
            if f.exists():
                old=pd.read_csv(f,index_col=0,parse_dates=True)
                df=pd.concat([old,df]).sort_index()
                df=df.loc[~df.index.duplicated(keep='last')]
            df.to_csv(f,float_format='%.10g')
        except Exception as exc:failure.append({'source':c,'error':str(exc)[:180]})
    if stock_getter is None:
        import yfinance as yf
        stock_getter=lambda t: yf.Ticker(t).history(period='10y',interval='1d',auto_adjust=True,timeout=22)
    # A transfer experiment runs on stock-only future evaluation slices.
    for s in _read_watchlist()[:6]:
        try:
            raw=clean_prices(stock_getter(s))
            if len(raw)<320:raise ValueError(f'주식 가격 자료가 {len(raw)}일뿐입니다.')
            result=transfer_audit(raw,crypto,horizon=21)
            result['symbol']=s
            reports.append(result)
        except Exception as exc:failure.append({'source':s,'error':str(exc)[:180]})
    payload={'timestamp_utc':now.isoformat(),'mode':'하루 1회 코인→주식 전이학습 실험',
             'status':'ok' if reports else 'failed',
             'integrated_into_live_stock_predictions':False,
             'why':'독립 검증이 추가로 필요하므로 자동으로 주식 모델을 교체하지 않습니다.',
             'results':reports,'errors':failure}
    _dump(root/'reports'/'crypto_transfer_latest.json',payload)
    print(json.dumps(payload,ensure_ascii=False))
    if not reports:raise RuntimeError('전이학습 실험을 실행하지 못했습니다. 데이터 연결을 확인하세요.')
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['hourly','daily'],required=True)
    mode=p.parse_args().mode
    (hourly if mode=='hourly' else daily)()

if __name__=='__main__':main()
