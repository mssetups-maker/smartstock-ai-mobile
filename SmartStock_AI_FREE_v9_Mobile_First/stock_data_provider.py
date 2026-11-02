"""Daily US equity history with clear failure provenance (research-only).

Yahoo Finance and Stooq are independent, best-effort public services. No API
availability, quote freshness, licensing, or adjustment consistency is promised.
Never substitute fabricated candles or stale cached prices on an outage.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import os
import re

import pandas as pd


class StockDataUnavailable(RuntimeError):
    pass


def _validate_structure(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f'{source}: 일봉이 비어 있습니다.')
    # do not repair contradictory prices; clean_prices checks them after
    # in-progress candles are removed by daily_market_state.
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) for col in df.columns]
    df.columns = [str(col).strip().title() for col in df.columns]
    required = ('Open', 'High', 'Low', 'Close', 'Volume')
    if not all(col in df.columns for col in required):
        raise ValueError(f'{source}: OHLCV 필수 열이 누락됐습니다.')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f'{source}: 날짜 인덱스가 없습니다.')
    if len(df) < 1:
        raise ValueError(f'{source}: 가격 기록이 없습니다.')
    df.attrs['stock_data_source'] = source
    return df


def _stooq_csv(symbol: str, *, now:datetime, timeout:float=12) -> pd.DataFrame:
    # Stooq CSV is an end-of-day source, not an intraday quote provider.
    # Some installations require a Stooq API key; unauthorized replies must
    # never be mistaken for price data.
    s=symbol.strip().upper()
    if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,14}',s) or s.startswith('^'):
        raise ValueError('Stooq 백업은 일반 미국주식/ETF 티커만 지원합니다.')
    if s.endswith(('.KS', '.KQ', '.T')):
        raise ValueError('Stooq 백업은 이 시장 티커를 지원하지 않습니다.')
    if not s.endswith('.US'):
        s=s+'.US'
    today=now.astimezone(timezone.utc).date()
    # Allow 11y for an approximate 10-year history, including holidays.
    start=today-timedelta(days=11*366)
    params={'s':s.lower(),'i':'d','d1':start.strftime('%Y%m%d'),
            'd2':today.strftime('%Y%m%d')}
    key=os.getenv('STOOQ_API_KEY','').strip()
    if key:params['apikey']=key
    url='https://stooq.com/q/d/l/?'+urlencode(params)
    req=Request(url,headers={'User-Agent':'SmartStockAI-Research/10.11.2'})
    with urlopen(req,timeout=timeout) as resp:
        data=resp.read(7_000_001)
    if len(data)>7_000_000:
        raise ValueError('Stooq 응답이 허용된 파일 크기를 넘었습니다.')
    if not data.lstrip().startswith(b'Date,Open,High,Low,Close,Volume'):
        raise ValueError('Stooq가 CSV 데이터를 제공하지 않았습니다(인증/사용량 제한/데이터 없음 가능).')
    frame=pd.read_csv(BytesIO(data))
    frame['Date']=pd.to_datetime(frame['Date'],errors='raise')
    return _validate_structure(frame.set_index('Date'),'Stooq EOD')


def fetch_stock_daily(symbol:str, *, now:datetime|None=None,
                      yahoo_getter=None, stooq_getter=None) -> pd.DataFrame:
    """Fetch daily candles, first Yahoo then optional Stooq public CSV.

    External outages are raised with both provider error summaries. Does NOT
    automatically promote stale cached data into current trading signals.
    """
    now=now or datetime.now(timezone.utc)
    errors=[]
    if yahoo_getter is None:
        import yfinance as yf
        yahoo_getter=lambda x: yf.Ticker(x).history(
            period='10y',interval='1d',auto_adjust=True,
            actions=False,timeout=16)
    try:
        return _validate_structure(yahoo_getter(symbol),'Yahoo adjusted daily')
    except Exception as exc:
        errors.append(f'Yahoo: {type(exc).__name__} {str(exc)[:120]}')
    if stooq_getter is None:
        stooq_getter=lambda x: _stooq_csv(x,now=now)
    try:
        return _validate_structure(stooq_getter(symbol),'Stooq EOD fallback')
    except Exception as exc:
        errors.append(f'Stooq: {type(exc).__name__} {str(exc)[:120]}')
    raise StockDataUnavailable(
        f'{symbol}: 무료 일봉 데이터 제공처 두 곳 모두 이용 불가. '
        +' | '.join(errors)
        +' — 실제 가격 없이 예측하지 않습니다.')
