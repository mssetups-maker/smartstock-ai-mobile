"""Free, explainable price-signal adaptation and non-predictive wave heuristics.

All learning labels use subsequent H complete bars and omit the most recent H.
The adaptive composite score is NOT a probability or a trading recommendation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import re
import numpy as np
import pandas as pd

from engine import HORIZON, swing_levels, validate_horizon

LABELS = {
    '이동평균 추세': '종가와 MA60, MA20/MA60 관계',
    'MACD': 'MACD 히스토그램 방향',
    'RSI 역추세': 'RSI 40 미만 반등·60 초과 과열 경계',
    '스토캐스틱 단기': 'Slow(5,3,3) %K와 %D',
    '스토캐스틱 중기': 'Slow(10,6,6) %K와 %D',
    '스토캐스틱 장기': 'Slow(20,12,12) %K와 %D',
    '가격 모멘텀': '최근 21거래일 수익률',
    '볼린저 평균회귀': '밴드 내 위치',
}


def candidate_signals(d:pd.DataFrame)->pd.DataFrame:
    """Discrete {-1,0,1} rules; causal indicators only."""
    c=d.Close
    rules={
        '이동평균 추세': np.sign((c-d.MA60)/d.MA60).where((d.MA20-d.MA60).abs()/d.MA60>.005, 0),
        'MACD':np.sign(d.MACD_hist),
        'RSI 역추세':pd.Series(np.select([d.RSI14<40,d.RSI14>60],[1,-1],default=0),index=d.index),
        '스토캐스틱 단기':np.sign(d.STO_K5-d.STO_D5),
        '스토캐스틱 중기':np.sign(d.STO_K10-d.STO_D10),
        '스토캐스틱 장기':np.sign(d.STO_K20-d.STO_D20),
        '가격 모멘텀':np.sign(c.pct_change(21)),
        '볼린저 평균회귀':pd.Series(np.select([c < d.BB_lower+(d.BB_upper-d.BB_lower)*.25,
                                          c > d.BB_lower+(d.BB_upper-d.BB_lower)*.75],[1,-1],default=0),index=d.index),
    }
    return pd.DataFrame(rules,index=d.index).replace([np.inf,-np.inf],np.nan)


@dataclass
class AdaptiveResult:
    score:float|None
    regime:str
    sample_count:int
    rows:list[dict]
    explanation:str


def adaptive_signals(d:pd.DataFrame, max_history:int=1050, horizon_days:int=HORIZON)->AdaptiveResult:
    """Historical rule weighting, without training on the current/future 21 bars.

    Evaluation points are spaced by the forecast horizon to avoid counting
    overlapping outcomes as independent. We don't call composite score a probability.
    """
    horizon_days = validate_horizon(horizon_days)
    c=d.Close
    latest=d.iloc[-1]
    atr_pct=float(latest.ATR14/latest.Close) if latest.Close>0 else 0
    adx=float(latest.ADX14) if np.isfinite(latest.ADX14) else 0
    regime=('강한 추세' if adx>=25 else '약한 추세·횡보')
    if atr_pct>.055: regime+=' / 고변동성'
    rules=candidate_signals(d)
    last_possible=len(d)-horizon_days-1
    indices=list(range(max(241,len(d)-max_history),max(241,last_possible)+1,horizon_days))
    if len(indices)<12:
        return AdaptiveResult(None,regime,len(indices),[], '학습 가능한 독립 표본이 부족합니다.')
    y=np.sign(c.shift(-horizon_days)/c-1)
    table=[]; weights=[]; current=[]
    for col in rules.columns:
        x=rules[col].iloc[indices]
        target=y.iloc[indices]
        eligible=x.notna() & target.notna() & x.ne(0) & target.ne(0)
        total=int(eligible.sum())
        if total>=8:
            win=float((x[eligible]==target[eligible]).mean())
            # Conservative shrinkage to chance; downweight rules worse than chance.
            adj=(win*total+0.5*24)/(total+24)
            w=max(0.05, (adj-.45)*5)
        else:
            win=None; w=0.05
        now=float(rules[col].iloc[-1]) if np.isfinite(rules[col].iloc[-1]) else 0.0
        table.append({'보조지표':col,'설명':LABELS[col], '과거 방향 적중률':win,
                      '평가 건수':total,'학습 가중치':round(w,3),
                      '현재 신호':'상승' if now>0 else '하락' if now<0 else '중립'})
        weights.append(w);current.append(now)
    if sum(weights)==0:return AdaptiveResult(None,regime,len(indices),table,'활성 지표가 없습니다.')
    score=float(100*np.dot(weights,current)/sum(weights))
    return AdaptiveResult(score,regime,len(indices),table,
                          f'과거 {horizon_days}거래일 방향에 따른 지표별 적중률을 수축 보정해 가중치를 계산합니다. '
                          '점수는 -100~+100 범위의 지표 투표이며 실제 상승확률이 아닙니다.')


def wave_candidates(d:pd.DataFrame, order:int=5, tail:int=360)->dict[str,Any]:
    """Confirmed swing structure only. Neither Fibonacci nor Elliott is uniquely identifiable."""
    x=d.tail(tail)
    lows,highs=swing_levels(x,order)
    points=sorted([(i,price,'저점') for i,price in lows]+[(i,price,'고점') for i,price in highs])
    # Resolve same-bar and successive same-type pivots into most extreme pivots.
    filtered=[]
    for ix,val,typ in points:
        if not filtered:filtered.append((ix,val,typ));continue
        prev=filtered[-1]
        if prev[2]==typ:
            if (typ=='고점' and val>prev[1]) or (typ=='저점' and val<prev[1]):filtered[-1]=(ix,val,typ)
        elif ix>prev[0]:filtered.append((ix,val,typ))
    recent=filtered[-9:]
    if len(recent)<4:return {'label':'판정 불가','notes':'확정된 고점·저점이 부족합니다.', 'points':[],'fib':{}}
    hi=[v for i,v,t in recent if t=='고점']
    lo=[v for i,v,t in recent if t=='저점']
    up=len(hi)>=2 and len(lo)>=2 and hi[-1]>hi[-2] and lo[-1]>lo[-2]
    down=len(hi)>=2 and len(lo)>=2 and hi[-1]<hi[-2] and lo[-1]<lo[-2]
    label='상승 스윙 구조' if up else '하락 스윙 구조' if down else '혼조 스윙 구조'
    bounds=d.tail(120)
    # fib drawn between last 120d high and low, clearly not a price forecast
    low=float(bounds.Low.min()); high=float(bounds.High.max())
    fib={f'{p:.1%}':round(high-(high-low)*p,2) for p in [0,.236,.382,.5,.618,.786,1]}
    rows=[{'날짜':x.index[i].strftime('%Y-%m-%d') if hasattr(x.index[i],'strftime') else str(x.index[i]),
           '구분':typ,'가격':round(float(val),3)} for i,val,typ in recent]
    return {'label':label,'notes':
            '확정된 스윙 고점·저점의 상승/하락 구조만 식별합니다. 엘리엇 1~5파·ABC 파동은 복수 해석이 가능하여 자동으로 확정하지 않습니다. '
            '최근 120일 피보나치 가격은 관측된 최고·최저 기반의 참고 수준입니다.',
            'points':rows,'fib':fib}


def free_explanation(symbol:str, d:pd.DataFrame, analysis:Any, adaptive:AdaptiveResult,
                     news:list[dict]|None=None, options:Any=None, fundamentals:dict|None=None,
                     market:dict|None=None, horizon_days:int=HORIZON)->str:
    """Deterministic Korean cross-source report; does not pretend to be a generative AI."""
    last=d.iloc[-1]
    lines=[f'### {symbol} 무료 종합 분석',
           f'- **기준:** {d.index[-1].strftime("%Y-%m-%d")} 확정 일봉 종가 {last.Close:,.2f}. 실시간 틱 데이터가 아닙니다.']
    if analysis.probability is None:
        lines.append('- **머신러닝:** 데이터 또는 독립 검증이 부족해 확률 표시를 보류했습니다.')
    else:
        lines.append(f'- **{horizon_days}거래일 상승 모델 추정:** {analysis.probability:.1%}. 과거 미보정 모델의 출력이며 실제 성공확률은 아닙니다.')
        if analysis.test_cases:
            lines.append(f'- **최근 검증:** {analysis.test_cases}개 비중복 구간, 방향 적중률 {analysis.backtest_accuracy:.1%}; 단순 다수 클래스 비교 {analysis.baseline_accuracy:.1%}.')
    lines.append(f'- **시장 상태:** {adaptive.regime}; 가중 신호 점수 {adaptive.score:+.1f}/100.' if adaptive.score is not None else f'- **시장 상태:** {adaptive.regime}. 학습 점수 산출 불가.')
    lines.append(f'- **기술:** RSI {last.RSI14:.1f}, ADX {last.ADX14:.1f}, MACD 히스토그램 {last.MACD_hist:+.3f}; 종가가 60일선 ' + ('위' if last.Close>last.MA60 else '아래') + '에 있습니다.')
    levels=analysis.levels
    lines.append(f'- **롱 가정 관리가격:** 손절 참고 {levels["stop"]:,.2f}, 1차 {levels["tp1"]:,.2f}, 2차 {levels["tp2"]:,.2f}. 동적 계산값이며 주문 가격이 아닙니다.')
    if options:
        r=lambda x: f'{x:.2f}' if x is not None and math.isfinite(float(x)) else '미확인'
        lines.append(f'- **옵션 스냅샷({options.expiry} 만기):** OI 풋/콜 {r(options.oi_pcr)}, IV {r(options.implied_vol)}, 예상 변동폭 {r(options.implied_move_pct)}(소수단위), Max Pain {r(options.max_pain)}. 데이터는 지연될 수 있으며 옵션 수급 방향을 확정할 수 없습니다.')
    else:lines.append('- **옵션:** 수집되지 않았습니다. 분석 결과에 옵션 데이터를 반영하지 않았습니다.')
    fin=fundamentals or {}
    if fin:
        fields=[f'{k}={v}' for k,v in list(fin.items())[:4]]
        lines.append('- **재무(제공처 기준):** '+', '.join(fields)+' — 공시 기준기간과 단위 확인 필요.')
    else: lines.append('- **재무:** 확인된 재무 수치를 입력하지 않았습니다.')
    if news:
        vals=[row.get('score',0) for row in news]
        lines.append(f'- **뉴스 제목 사전 점수:** 긍정 {sum(v>0 for v in vals)}건, 부정 {sum(v<0 for v in vals)}건, 중립·미판정 {sum(v==0 for v in vals)}건. 기사 본문 분석이나 주가 예측으로 간주하지 않습니다.')
    else:lines.append('- **뉴스:** 분석 가능한 제목을 확보하지 못했습니다.')
    if market:
        lines.append('- **시장 비교:** '+', '.join(f'{k} 최근5일 {v.get("five_day_pct",0):+.1f}%' for k,v in list(market.items())[:3]))
    lines.append('**판단 원칙:** 신호가 서로 다르거나 검증 성능이 낮으면 관망으로 해석해야 합니다. 수치만으로 매수 결정을 내리지 마세요.')
    return '\n\n'.join(lines)
