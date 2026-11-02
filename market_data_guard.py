"""Integrity checks and conservative completed-daily-bar identification.

The cutoff establishes eligibility by clock time only; it cannot certify an
exchange session/holiday, a vendor's final revision or point-in-time integrity.
Never use this to infer that an actively trading daily bar has closed.
"""
from __future__ import annotations
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd


def check_ohlcv(frame: pd.DataFrame) -> None:
    """Reject impossible OHLCV input rather than silently train on distorted candles."""
    required={'Open','High','Low','Close','Volume'}
    if not isinstance(frame,pd.DataFrame) or frame.empty or not required.issubset(frame.columns):
        raise ValueError('가격 데이터가 비어 있거나 OHLCV 열이 누락됐습니다.')
    vals=frame[['Open','High','Low','Close']].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals<=0).any():
        raise ValueError('OHLC가 누락되거나 0 이하/무한대인 캔들이 있습니다. 분석을 중단합니다.')
    o,h,l,c=vals.T
    invalid=(h<np.maximum.reduce([o,l,c])-1e-9) | (l>np.minimum.reduce([o,h,c])+1e-9)
    if invalid.any():
        examples=[str(i)[:19] for i in frame.index[np.flatnonzero(invalid)[:3]]]
        raise ValueError('OHLC 관계 오류가 있는 캔들 '+str(int(invalid.sum()))+'개: '+', '.join(examples)+'; 지표 계산을 중단합니다.')
    vol=pd.to_numeric(frame['Volume'],errors='coerce').to_numpy(dtype=float)
    if np.isinf(vol).any() or np.any(np.isfinite(vol)&(vol<0)):
        raise ValueError('거래량이 음수 또는 무한대인 데이터가 있습니다.')


def daily_market_state(frame:pd.DataFrame,symbol:str,now:datetime|None=None) -> tuple[pd.DataFrame,str]:
    """Exclude a same-market-date daily bar before closing grace period.

    Applies to equities on supported US/KR markets, not cryptocurrency.
    A Yahoo daily index may be timezone-aware; normalize to exchange-local
    session date before comparison.  Does not guess non-trading holidays.
    """
    if now is None:now=datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('now는 시간대를 포함해야 합니다.')
    kr=str(symbol).upper().endswith(('.KS','.KQ'))
    tz=ZoneInfo('Asia/Seoul' if kr else 'America/New_York')
    close=time(15,45) if kr else time(16,30)
    local=now.astimezone(tz)
    if frame.empty:return frame,'데이터 없음'
    data=frame.copy()
    if isinstance(data.index,pd.DatetimeIndex):
        idx=data.index.tz_convert(tz).tz_localize(None) if data.index.tz is not None else data.index
        data.index=idx
    data=data.sort_index()
    last_date=pd.Timestamp(data.index[-1]).date()
    if last_date>local.date():
        raise ValueError('시세 기준일이 해당 거래소의 오늘보다 미래입니다.')
    if last_date==local.date() and local.time()<close:
        data=data.loc[pd.to_datetime(data.index).date<local.date()].copy()
        if data.empty:raise ValueError('현재는 미완성 일봉만 있습니다. 확정 일봉이 도착한 후 다시 분석하세요.')
        return data,('오늘의 일봉은 장 종료 전이므로 제외했습니다. '+
                    str(data.index[-1].date())+'까지의 완료된 봉만 사용합니다. 공급자 수정 가능성은 남습니다.')
    return data,('거래소 마감시각 기준 완료 가능 일봉입니다. 공급처 지연·수정 가능성을 확인하세요.')
