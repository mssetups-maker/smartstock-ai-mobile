"""Causal OHLCV/technical-indicator verification and alternative scenario search.

All interpretations are *rule diagnostics*, not guaranteed future outcomes.
Never silently turn a contradictory or missing value into a bullish label.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from engine import indicators
from market_data_guard import check_ohlcv

PRICE_FEATURES = ('MA20','MA60','MA120','MA240','EMA12','EMA26','MACD','MACD_signal',
                  'MACD_hist','ATR14','ADX14','RSI14','BB_upper','BB_lower',
                  'STO_K5','STO_D5','STO_K10','STO_D10','STO_K20','STO_D20',
                  'MFI14','OBV','OBV_slope20')
RANGED = ('RSI14','ADX14','MFI14','STO_K5','STO_D5','STO_K10','STO_D10','STO_K20','STO_D20')


def verify_and_rebuild(raw: pd.DataFrame, supplied: pd.DataFrame | None = None) -> dict[str,Any]:
    """Recompute authoritative indicators; reject corrupt candles and impossible metrics.

    A stale/corrupted supplied indicator is **never** used for candidate ranking.
    All checks are at the latest fully supplied bar and use only historical bars.
    """
    check_ohlcv(raw)
    if not isinstance(raw.index,pd.DatetimeIndex):
        raise ValueError('가격 날짜 인덱스를 확인하세요.')
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError('중복·역순 가격 날짜를 정리한 뒤 분석해야 합니다.')
    reference=indicators(raw)
    if len(reference)<90:raise ValueError('검증 가능한 지표 기간이 부족합니다.')
    last=reference.iloc[-1]
    blocked=[];skipped=[];repaired=[]
    for col in PRICE_FEATURES:
        if col not in reference:
            blocked.append(f'{col}: 계산 항목 누락')
            continue
        v=last[col]
        if pd.isna(v):
            skipped.append(f'{col}: 현 시점 계산 자료 부족/가격 변화 또는 거래량 부족')
            continue
        if not math.isfinite(float(v)):
            blocked.append(f'{col}: 비정상 숫자')
            continue
        if col in RANGED and not -1e-6<=float(v)<=100+1e-6:
            blocked.append(f'{col}: 0~100 범위 위반')
        if col=='ATR14' and float(v)<0:blocked.append('ATR14: 음수')
        if supplied is not None:
            if col not in supplied or supplied.empty or pd.isna(supplied[col].iloc[-1]):
                repaired.append(f'{col}: 재산정')
                continue
            older=float(supplied[col].iloc[-1]);fresh=float(v)
            if not math.isfinite(older) or not math.isclose(older,fresh,rel_tol=1e-5,abs_tol=1e-7):
                repaired.append(f'{col}: 이전 값 불일치로 재산정')
    macd=float(last['MACD']);sig=float(last['MACD_signal']);hist=float(last['MACD_hist'])
    if not math.isclose(hist,macd-sig,rel_tol=1e-7,abs_tol=1e-8):
        blocked.append('MACD 히스토그램 일관성 오류')
    if last['BB_upper'] < last['BB_lower']:
        blocked.append('볼린저 상하단 관계 오류')
    vol=raw['Volume'].tail(20)
    volume_reliable=bool((vol>0).sum()>=10)
    if not volume_reliable:
        skipped.append('최근 거래량 불충분: MFI·OBV 판단에서 제외')
    if blocked:
        return {'status':'차단','df':reference,'issues':blocked,'skipped':skipped,'repaired':repaired,
                'volume_reliable':volume_reliable}
    return {'status':'재산정' if repaired else '정상','df':reference,'issues':[],
            'skipped':skipped,'repaired':repaired,'volume_reliable':volume_reliable}


def _vote(value, low, high):
    if value is None or pd.isna(value) or not math.isfinite(float(value)):return None
    return 1 if float(value)>high else -1 if float(value)<low else 0


def latest_signal(frame: pd.DataFrame, volume_reliable: bool = True, row: int = -1) -> dict[str,Any]:
    """Causal fixed-rule vote from the selected bar; no fitted coefficients.

    Values -1/0/1 are directional *votes* not probabilities. The fixed coarse
    bucket permits comparable prospective-style historical replay.
    """
    if frame.empty:return {'bucket':'unavailable','bias':0.,'votes':[], 'available':0,'direction':'판단 보류'}
    s=frame.iloc[row];close=float(s['Close']);votes=[]
    def add(label,value):
        if value is not None:
            votes.append({'indicator':label,'vote':int(value)})
    if pd.notna(s.get('MA60')):
        add('60일 이동평균 추세',_vote(close / float(s['MA60']) - 1,-.005,.005))
    if pd.notna(s.get('MA20')) and pd.notna(s.get('MA60')):
        add('20/60일 이동평균 배열',_vote(float(s['MA20']) / float(s['MA60']) - 1,-.003,.003))
    if pd.notna(s.get('MA120')) and pd.notna(s.get('MA60')):
        add('60/120일 이동평균 배열',_vote(float(s['MA60']) / float(s['MA120']) - 1,-.003,.003))
    add('RSI(14)',_vote(s.get('RSI14'),45,55))
    if pd.notna(s.get('MACD_hist')):
        add('MACD 히스토그램',_vote(float(s['MACD_hist']) / close,-.0003,.0003))
    if pd.notna(s.get('STO_K20')) and pd.notna(s.get('STO_D20')):
        add('슬로우 스토캐스틱(20)',_vote(float(s['STO_K20'])-float(s['STO_D20']),-4,4))
    if volume_reliable:
        add('MFI(14)',_vote(s.get('MFI14'),45,55))
        if pd.notna(s.get('OBV_slope20')):
            # A signed OBV slope provides a vote, not an estimated return size.
            add('OBV 거래량 추세',_vote(s['OBV_slope20'],-1e-10,1e-10))
    n=len(votes)
    total=sum(v['vote'] for v in votes)
    bias=total/n if n else 0.
    bucket = 'bull' if n>=4 and bias>=.25 else 'bear' if n>=4 and bias<=-.25 else 'mixed' if n>=4 else 'unavailable'
    direction={'bull':'상승 우세','bear':'하락 우세','mixed':'혼조/박스 가능성','unavailable':'판단 보류'}[bucket]
    return {'bucket':bucket,'bias':bias,'votes':votes,'available':n,'direction':direction,
            'conflicts':sum(v['vote']!=0 and v['vote']*bias<0 for v in votes)}


def scenario_candidates(signal: dict[str,Any], model_direction_score:float|None=None) -> list[dict]:
    """Three independently defined possibilities. Rank scores never denote probability."""
    if signal['available']<4:return []
    x=float(signal['bias']);ml=None
    if (model_direction_score is not None and not isinstance(model_direction_score,bool)
            and np.isfinite(model_direction_score) and 0<=model_direction_score<=1):
        # The ML score is a secondary rank signal, not a calibrated forecast.
        ml=20*(float(model_direction_score)-.5)
    else:ml=0.
    # The mixed rank reflects directional disagreement, not 'sideways chances'.
    values={'상승':50+38*x+ml,'하락':50-38*x-ml,'횡보':50+12*(1-abs(x))}
    details=[f'{v["indicator"]}: '+('상승' if v['vote']>0 else '하락' if v['vote']<0 else '중립')
             for v in signal['votes']]
    result=[]
    for outcome in ('상승','하락','횡보'):
        result.append({'id':'rule-'+{'상승':'up','하락':'down','횡보':'side'}[outcome],
                       'title':f'{outcome} 방향 해석 · 기술지표 교차 검증',
                       'valid':True,'data_valid':True,'outcome':outcome,
                       'rank_score':max(0.,min(100.,float(values[outcome]))),
                       'evidence':details[:8],
                       'invalidation':'지표 갱신 또는 가격 구조가 변화하면 자동 재산정',
                       'origin':'고정 지표 교차 확인'})
    return result


def indicator_table(signal:dict[str,Any]) -> pd.DataFrame:
    return pd.DataFrame([{'지표':v['indicator'], '상태':('상승' if v['vote']>0 else '하락' if v['vote']<0 else '중립')}
                         for v in signal.get('votes',[])],columns=['지표','상태'])
