"""Transparent opportunity funnel for COMPLETED daily bars.

Never emits trade orders. The exact gate conditions reuse market_decision._signals
and are kept in lock-step with its three price-action strategies. Historical
counts are *bar signal days*, not independent trades or profitable outcomes.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import os
import tempfile

import pandas as pd
import numpy as np
from engine import clean_prices
from market_decision import _features, _signals, STRATEGIES
from paper_audit import _asof_frame

VERSION='10.10-opportunity-funnel-1'


def _rule_checks(row:pd.Series) -> dict[str,dict[str,bool]]:
    c=float(row.Close);atr=float(row.atr14);ma20=float(row.ma20)
    trend=c>float(row.ma60) and ma20>float(row.ma60)
    return {
        'breakout': {
            '상승 추세(종가>60일선, 20일선>60일선)':trend,
            '직전 20봉 고점 돌파':c>float(row.prior_high20),
            '거래량 >= 직전 20봉 평균의 1.1배':float(row.volratio)>=1.1,
            'RSI 50~85':50<=float(row.rsi)<=85},
        'pullback':{
            '상승 추세(종가>60일선, 20일선>60일선)':trend,
            '20일선 주변의 눌림목':ma20-.6*atr<=c<=ma20+.8*atr,
            '전일 종가 대비 상승':c>float(row.prior_close),
            '양봉/도지(종가>=시가)':c>=float(row.Open),
            'RSI 40~72':40<=float(row.rsi)<=72},
        'rebound':{
            '20봉 저점으로부터 0~1.3 ATR':0<=c-float(row.prior_low20)<=1.3*atr,
            '전일 종가 대비 상승':c>float(row.prior_close),
            'RSI 전일 대비 개선':float(row.rsi)>float(row.prior_rsi),
            '양봉/도지(종가>=시가)':c>=float(row.Open),
            'RSI 22~56':22<=float(row.rsi)<=56,
            '5일 이동평균 위':c>float(row.ma5)}
    }


def diagnose_one(prices:pd.DataFrame,symbol:str,*,now:datetime|None=None,
                 lookback:int=252) -> dict[str,Any]:
    """Return as-of diagnostics; reject unclosed candles and malformed prices.

    No new signal or profitable return is computed from hypothetical future
    candles. Counts are not walk-forward portfolio returns.
    """
    now=now or datetime.now(timezone.utc)
    if not symbol or len(symbol)>32:raise ValueError('종목명이 유효하지 않습니다.')
    if not 20<=lookback<=3000:raise ValueError('분석 기간이 유효하지 않습니다.')
    frame=_asof_frame(clean_prices(prices),now)
    if len(frame)<90:raise ValueError('90개 이상의 완료 일봉이 필요합니다.')
    f=_signals(_features(frame));last=f.iloc[-1]
    if not np.isfinite(last[['Close','ma20','ma60','atr14','rsi','volratio','prior_low20','prior_high20']].to_numpy(dtype=float)).all():
        raise ValueError('필요한 기술지표가 완성되지 않았습니다.')
    checks=_rule_checks(last)
    results={}
    for key,labels in checks.items():
        pass_count=sum(labels.values());num=len(labels)
        exact=bool(last[key])
        # Guaranteed match with production strategy detection: fail closed.
        if exact != (pass_count==num):
            raise ValueError(f'{key}: 매매 규칙과 진단 규칙이 불일치합니다. 분석 중단.')
        recent=f.tail(lookback)
        signals=recent.index[recent[key]]
        results[key]={
            'name':STRATEGIES[key], 'signal_today':exact,
            'passed':pass_count,'total':num,
            'missing':[name for name,ok in labels.items() if not ok],
            'checks':{name:bool(ok) for name,ok in labels.items()},
            'signal_days_in_lookback':int(recent[key].sum()),
            'last_signal_date':str(pd.Timestamp(signals[-1]).date()) if len(signals) else None,
            'is_near_miss':not exact and pass_count>=num-1,
            'historical_count_is_trades':False,
        }
    selected=sorted(results.items(),key=lambda p:(not p[1]['signal_today'],
                                -(p[1]['passed']/p[1]['total']),p[0]))[0][0]
    signal=any(r['signal_today'] for r in results.values())
    close=float(last.Close)
    # Explanatory thresholds are NOT limit orders or approved prices.
    levels={
        'prior_20_day_high':round(float(last.prior_high20),6),
        'prior_20_day_low':round(float(last.prior_low20),6),
        'ma20':round(float(last.ma20),6),
        'ma60':round(float(last.ma60),6),
        'atr14':round(float(last.atr14),6),
        'breakout_gap_pct':round((float(last.prior_high20)/close-1)*100,2),
        'pullback_band_low':round(float(last.ma20)-.6*float(last.atr14),6),
        'pullback_band_high':round(float(last.ma20)+.8*float(last.atr14),6),
    }
    return {
        'version':VERSION,'symbol':symbol.upper(),
        'bar_date':str(pd.Timestamp(f.index[-1]).date()),
        'close':round(close,6),'observed_candles':len(frame),
        'period_bars':min(len(f),lookback),'source_scope':'완성 일봉 가격·거래량만',
        'signal_today':signal,
        'classification':'규칙 기반 전략 후보(매수 승인 아님)' if signal else '해당 전략 매수 신호 없음',
        'closest_strategy':selected, 'strategies':results,
        'levels':levels,'paper_only':True,
        'reliability_note':'과거 발생일 수는 독립 체결 거래나 수익률이 아닙니다. 시장 위험, 뉴스/옵션, 실시간 변동은 반영하지 않았습니다.'}


def batch_diagnostics(frames:dict[str,pd.DataFrame], *, now:datetime|None=None,
                       lookback:int=252) -> dict[str,Any]:
    now=now or datetime.now(timezone.utc)
    items={};errors=[]
    for sym,frame in sorted(frames.items()):
        try:items[sym.upper()]=diagnose_one(frame,sym,now=now,lookback=lookback)
        except Exception as exc:
            errors.append({'symbol':sym,'error':str(exc)[:200]})
    active=sum(a['signal_today'] for a in items.values())
    near=sum(not a['signal_today'] and any(s['is_near_miss'] for s in a['strategies'].values()) for a in items.values())
    return {
        'version':VERSION,'timestamp_utc':now.astimezone(timezone.utc).isoformat(),
        'status':'ok' if items and not errors else 'partial' if items else 'failed',
        'covered_symbols':len(items),'signal_symbols':active,
        'near_miss_symbols':near,'inactive_symbols':len(items)-active,
        'errors':errors,'by_symbol':items,'paper_only':True,
        'warning':'후보가 많아지는 것 자체는 정확도 향상이 아닙니다. 규칙 진입 및 보유 기간별 실전 이후 성적을 따로 검증해야 합니다.'}


def save_diagnostics(path:Path,report:dict[str,Any]):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    # Do not churn the Git repository on a rerun of unchanged results.
    previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    previous_cmp={k:v for k,v in previous.items() if k!='timestamp_utc'}
    current_cmp={k:v for k,v in report.items() if k!='timestamp_utc'}
    if previous_cmp==current_cmp:return False
    fd,tmp=tempfile.mkstemp(prefix='.opportunity_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as out:out.write(content)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.remove(tmp)
    return True
