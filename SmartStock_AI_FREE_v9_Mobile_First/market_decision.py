"""SmartStock 10.7: transparent strategy scouting, paper-only gates and risk plan.

Research, NOT investment advice or trading automation.  All candidate formation
uses information known at the signal close.  Historical experiments enter at
NEXT bar's open; no prior-close fills.  ATR stops are simulated with worst-case
stop-first intrabar execution and opening-gap slippage.  Historical experiments
are illustrative, not point-in-time ML walk-forward predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Any
import numpy as np
import pandas as pd

PROFILES = {
    '적극형': {'risk_pct': .75, 'rr_min': 1.5, 'max_alloc': 20., 'min_sample': 10},
    '균형형': {'risk_pct': .50, 'rr_min': 1.8, 'max_alloc': 20., 'min_sample': 12},
    '보수형': {'risk_pct': .25, 'rr_min': 2.0, 'max_alloc': 15., 'min_sample': 14},
}
LEVERAGED = {'TSLL', 'TQQQ', 'SOXL', 'SQQQ', 'SPXL', 'UPRO', 'SOXS', 'FNGU', 'NVDL', 'TSLQ'}
STRATEGIES = {'breakout': '상방 돌파', 'pullback': '상승 눌림목', 'rebound': '단기 반등'}


@dataclass
class StrategyStats:
    key: str
    name: str
    occurrences: int
    training_count: int
    holdout_count: int
    avg_net: float | None
    holdout_avg_net: float | None
    positive_rate: float | None
    holdout_positive_rate: float | None
    passed: bool
    reason: str


@dataclass
class TradePlan:
    stage: str
    strategy_key: str | None
    strategy_name: str
    candidate: bool
    score: float | None
    verified: bool
    reasons: list[str]
    blocks: list[str]
    entry: float | None
    stop: float | None
    target1: float | None
    target2: float | None
    reward_risk: float | None
    risk_pct: float
    max_alloc_pct: float
    stop_risk_pct: float | None
    stats: list[StrategyStats]
    last_bar: str
    profile: str


def _features(prices: pd.DataFrame) -> pd.DataFrame:
    """Price and volume data only; all rolling windows look backwards."""
    base = prices[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).copy().sort_index()
    c, o, hi, lo, v = (base[x] for x in ('Close','Open','High','Low','Volume'))
    for n in (5,20,60):
        base[f'ma{n}'] = c.rolling(n, min_periods=n).mean()
    base['prior_high20'] = hi.shift(1).rolling(20, min_periods=20).max()
    base['prior_low20'] = lo.shift(1).rolling(20, min_periods=20).min()
    base['volratio'] = v / v.shift(1).rolling(20, min_periods=20).mean().replace(0,np.nan)
    tr=pd.concat((hi-lo,(hi-c.shift()).abs(),(lo-c.shift()).abs()), axis=1).max(axis=1)
    base['atr14'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    delta=c.diff()
    up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    base['rsi'] = 100-100/(1+up/down.replace(0,np.nan))
    base.loc[(down==0)&(up>0), 'rsi']=100
    base.loc[(up==0)&(down>0), 'rsi']=0
    base['prior_close']=c.shift(1)
    base['prior_rsi']=base.rsi.shift(1)
    return base


def _signals(f:pd.DataFrame) -> pd.DataFrame:
    c=f.Close; ma20=f.ma20; ma60=f.ma60; atr=f.atr14
    trend=(c>ma60)&(ma20>ma60)
    f=f.copy()
    f['breakout']=(trend & (c>f.prior_high20) & (f.volratio>=1.1) & (f.rsi>=50) & (f.rsi<=85))
    f['pullback']=(trend & (c>=ma20-0.6*atr) & (c<=ma20+0.8*atr) &
                   (c>f.prior_close) & (c>=f.Open) & f.rsi.between(40,72))
    near_low=(c-f.prior_low20).between(0,1.3*atr)
    f['rebound']=(near_low & (c>f.prior_close) & (f.rsi>f.prior_rsi) &
                  (c>=f.Open) & f.rsi.between(22,56) & (c>f.ma5))
    for k in STRATEGIES:
        f[k]=f[k].fillna(False).astype(bool)
    return f


def _levels(row:pd.Series, rr_min:float)->tuple[float,float,float,float]:
    entry=float(row.Close); atr=float(row.atr14)
    atr=max(atr,entry*.002)
    recent_low=float(row.prior_low20) if isfinite(float(row.prior_low20)) else entry-2*atr
    # Cap wide stops to avoid unbounded exposure, without pretending a closer
    # level is technically justified.  A too-wide stop yields a non-candidate.
    stop=min(entry-1.5*atr, recent_low-0.15*atr) if entry-recent_low <= 2.8*atr else entry-1.7*atr
    stop=max(stop,entry*.65)
    distance=entry-stop
    target=entry+rr_min*distance
    return entry,stop,target,entry+(rr_min+0.8)*distance


def _simulate(prices:pd.DataFrame, f:pd.DataFrame, k:str, horizon:int,
              rr:float,cost_pct:float=.3)->list[dict[str,Any]]:
    """One paper position at a time. Re-arm after ACTUAL exit, not full horizon.

    Signal at close t -> entry at next open.  A trade exiting before its
    holding horizon can be evaluated without needing future unseen bars.
    Unresolved trades at the end are NOT counted as wins or losses.
    Same bar touching TP and stop exits at stop; gaps use opening prices.
    """
    n=len(f);out=[];t=65
    while t<n-1:
        if not bool(f[k].iat[t]):
            t+=1;continue
        row=f.iloc[t]
        if not np.isfinite(row[['Close','atr14','prior_low20']].to_numpy(dtype=float)).all():
            t+=1;continue
        _,stop,tp,_=_levels(row,rr)
        actual_entry=float(prices.Open.iloc[t+1])
        # If the planned stop or target has already been bypassed by the gap,
        # this is no longer the proposed trade. Avoid fictional fills.
        if not np.isfinite(actual_entry) or actual_entry<=stop or actual_entry>=tp:
            t+=1;continue
        end=min(n-1,t+horizon)
        realized=None;kind=None;exit_i=None
        for ix in range(t+1,end+1):
            bar=prices.iloc[ix]
            open_=float(bar.Open);low=float(bar.Low); high=float(bar.High)
            if open_<=stop:realized=open_;kind='시가 갭 손절';exit_i=ix;break
            if open_>=tp:realized=open_;kind='시가 갭 목표';exit_i=ix;break
            if low<=stop:realized=stop;kind='손절(동시 터치 시 우선)';exit_i=ix;break
            if high>=tp:realized=tp;kind='목표';exit_i=ix;break
        if realized is None:
            if t+horizon>n-1:
                t+=1;continue # Not enough future observations to settle
            realized=float(prices.Close.iloc[t+horizon]);kind='기간 만료';exit_i=t+horizon
        ret=realized/actual_entry-1-cost_pct/100
        out.append({'signal_i':t,'date':str(f.index[t].date()),'entry':actual_entry,
                    'net':float(ret),'exit_type':kind,'exit_i':exit_i,
                    'exit_date':str(f.index[exit_i].date())})
        # A closed trade frees capital. Scan from the NEXT completed bar.
        # This 1-bar cooldown is conservative and avoids same-bar reuse.
        t=max(t+1,exit_i+1)
    return out


def _stats(prices:pd.DataFrame,f:pd.DataFrame, horizon:int,profile:str)->list[StrategyStats]:
    p=PROFILES[profile];result=[]
    for key,name in STRATEGIES.items():
        events=_simulate(prices,f,key,horizon,p['rr_min'])
        # Time-ordered split; if fewer than 2 events, there's no holdout.
        cut=max(1,int(len(events)*.6)) if len(events)>=2 else len(events)
        dev=events[:cut];test=events[cut:]
        avg=lambda rows:float(np.mean([e['net'] for e in rows])) if rows else None
        hit=lambda rows:float(np.mean([e['net']>0 for e in rows])) if rows else None
        devavg=avg(dev);holdavg=avg(test)
        enough=(len(events)>=p['min_sample'] and len(test)>=4)
        passed=bool(enough and devavg is not None and holdavg is not None and
                    devavg>0 and holdavg>0 and hit(test)>=.5)
        if not enough:
            reason=f'독립 사례 {len(events)}건 / 최소 {p["min_sample"]}건, 뒤쪽 평가 {len(test)}건 / 최소 4건'
        elif not passed:reason='과거 앞·뒤 구간 평균 순수익 또는 후반 승률 기준 미달'
        else:reason='과거 앞·뒤 구간 참고 기준 통과(실전 수익 보장 아님)'
        result.append(StrategyStats(key,name,len(events),len(dev),len(test),avg(events),
                                    holdavg,hit(events),hit(test),passed,reason))
    return result


def evaluate_market(prices:pd.DataFrame, *, horizon:int=21, score:float|None=None,
                    strict_recommendation:str='판단 보류', quality:str='정상',
                    symbol:str='', profile:str='균형형', hypothetical:bool=False)->TradePlan:
    """Return descriptive stage; never transform unverified signal into live order.

    '시험 매수' is explicitly PAPER-ONLY. A strict-audit success is required
    for '적극 매수 검토', plus separate strategy evidence and quality checks.
    """
    if profile not in PROFILES:raise ValueError('지원하지 않는 위험 성향입니다.')
    if not 1<=horizon<=252:raise ValueError('예측 기간은 1~252 거래일이어야 합니다.')
    f=_signals(_features(prices))
    if len(f)<90:raise ValueError('최근 가격 데이터가 90거래일 미만입니다.')
    last=f.iloc[-1];close=float(last.Close)
    safe_score=float(score) if score is not None and np.isfinite(score) and 0<=score<=1 else None
    p=PROFILES[profile]
    blocks=[]
    if hypothetical:blocks.append('가상·업로드 시세로 실제 매수 판단 불가')
    if quality=='차단':blocks.append('일봉 데이터 품질 차단')
    elif quality=='주의':blocks.append('시세 품질 주의: 실시간 체결 판단 불가')
    if safe_score is None:blocks.append('AI 방향점수 없음')
    if not np.isfinite(last[['Close','ma20','ma60','atr14','rsi']].to_numpy(dtype=float)).all():
        blocks.append('현재 기술지표 결측')
    leverage=symbol.upper() in LEVERAGED
    risk_pct=min(p['risk_pct'],.25) if leverage else p['risk_pct']
    max_alloc=min(p['max_alloc'],10.) if leverage else p['max_alloc']
    if leverage:blocks.append('레버리지·인버스 ETF: 일반 종목보다 큰 위험 제한 필요')
    stats=_stats(prices,f,horizon,profile)
    rank={'breakout':0,'pullback':1,'rebound':2}
    live=[s for s in stats if bool(last[s.key])]
    live.sort(key=lambda s:(not s.passed,rank[s.key]))
    chosen=live[0] if live else None
    reasons=[]
    trend=bool(close>last.ma60 and last.ma20>last.ma60)
    if chosen:
        reasons.append(f'현재 {chosen.name} 조건 감지(확정 일봉 기준)')
        reasons.append(f'과거 독립 사례 {chosen.occurrences}건·검증 후반 {chosen.holdout_count}건')
        reasons.append(chosen.reason)
    else:reasons.append('현재 돌파·눌림목·반등의 확정 일봉 조건에 해당하지 않음')
    if safe_score is not None:
        reasons.append(f'모델 방향점수 {safe_score*100:.1f}점(보정된 확률 아님)')
    reasons.append('전략 자체의 과거 수익은 검증 통계일 뿐 미래 성과를 보장하지 않음')
    serious=quality=='차단' or hypothetical
    if serious:
        stage='관망';reasons.insert(0,'가격 정보가 시연용이거나 품질 차단이므로 실제 투자 판단 제공 안 함')
    elif chosen and chosen.passed and strict_recommendation=='조건부 매수 검토' and safe_score is not None and safe_score>=.68 and not leverage and quality=='정상':
        stage='적극 매수 검토'
    elif chosen and safe_score is not None and safe_score>=.35 and quality in ('정상','주의'):
        stage='시험 매수 후보(모의매매 전용)'
    elif (safe_score is not None and safe_score<.32 and not trend):
        stage='위험 회피'
    elif trend or (isfinite(float(last.prior_high20)) and close>float(last.prior_high20)-1.5*float(last.atr14)):
        stage='매수 대기'
    else:stage='관망'
    if stage=='시험 매수 후보(모의매매 전용)':
        reasons.append('실전 매수 승인 아님: 먼저 모의매매에 기록하고 검증 필요')
        if quality=='주의':reasons.append('신규 상장·시세 품질 경고가 있어 실전 진입은 권하지 않음')
    if stage=='매수 대기':
        reasons.append('종가 확정 시 전략 신호가 나오는지 관찰. 조건 미충족 상태에서 추격하지 않음')
    entry=stop=t1=t2=rr=dist=None
    if chosen and not serious:
        entry,stop,t1,t2=_levels(last,p['rr_min'])
        rr=(t1-entry)/(entry-stop)
        dist=(entry-stop)/entry*100
        if dist>12:
            reasons.append(f'손절까지 {dist:.1f}%: 변동성이 높아 진입 규모 매우 작게 산정해야 함')
    return TradePlan(stage,chosen.key if chosen else None,chosen.name if chosen else '확정 전략 없음',
                     bool(chosen),safe_score,bool(chosen and chosen.passed and quality=='정상'),
                     reasons,blocks,entry,stop,t1,t2,rr,risk_pct,max_alloc,dist,stats,
                     str(pd.Timestamp(f.index[-1]).date()),profile)


def example_position(plan:TradePlan, portfolio:float, round_trip_cost_pct:float=.3)->dict[str,float|int|str]:
    """Illustrative sizing only.  Accounts for fees in risk budget.
    The actual gap/downside may exceed the planned amount.
    """
    if portfolio<=0 or not isfinite(portfolio) or not 0<=round_trip_cost_pct<10:
        raise ValueError('투자금·비용 범위를 확인하세요.')
    if plan.entry is None or plan.stop is None or plan.stop>=plan.entry:
        return {'shares':0,'reason':'유효한 진입·손절 계획 없음'}
    risk_share=plan.entry-plan.stop+plan.entry*round_trip_cost_pct/100
    risk_budget=portfolio*plan.risk_pct/100
    max_risk_qty=floor(risk_budget/risk_share)
    max_alloc_qty=floor(portfolio*plan.max_alloc_pct/100/plan.entry)
    qty=max(0,min(max_risk_qty,max_alloc_qty))
    return {'shares':qty,'position_value':qty*plan.entry,
            'estimated_loss_if_stop_fills':qty*risk_share,
            'risk_budget':risk_budget, 'max_alloc_pct':plan.max_alloc_pct,
            'cost_pct':round_trip_cost_pct,
            'reason':'갭 하락·체결 슬리피지는 별도. 실거래 지시가 아닌 가상 위험 계산'}
