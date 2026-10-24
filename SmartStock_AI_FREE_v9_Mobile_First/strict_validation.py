"""Conservative chronological out-of-sample gate for stock-price predictions.

No backtest can guarantee future correctness. Every test-day model is fitted
only on bars whose forward H-day outcome was already observed at that day.
Fixed selection thresholds; no parameter optimization on the holdout.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from engine import FEATURES, indicators, make_features, validate_horizon

BUY_THRESHOLD = 0.68
AVOID_THRESHOLD = 0.32
MIN_TESTS = 18
MIN_SIGNALS = 10
COST_FRACTION = .003  # illustrative round-trip friction (30bp)


def wilson_lower(wins: int, total: int, z: float = 1.96) -> float | None:
    if total == 0:
        return None
    p = wins / total
    den = 1+z*z/total
    return (p+z*z/(2*total)-z*sqrt((p*(1-p)+z*z/(4*total))/total))/den


def _model(n: int) -> RandomForestClassifier:
    """Fixed before examining the test outcomes; no holdout hyperparameter search."""
    return RandomForestClassifier(n_estimators=68, max_depth=5,
          min_samples_leaf=max(8, n//250), max_features=.7, random_state=37,
          class_weight='balanced_subsample',n_jobs=-1)


def prediction_at(prices: pd.DataFrame, test_position: int, horizon_days: int) -> tuple[float, int, float] | None:
    """Fit solely on known labels strictly earlier than test date.

    Retaining only rows 0..t-h-1 prevents training labels from using the test
    date or any later prices. Feature computation uses past/current OHLCV only.
    """
    h = validate_horizon(horizon_days)
    if not 0 <= test_position < len(prices) or test_position < 400+h:
        return None
    d=indicators(prices.iloc[:test_position+1])
    x=make_features(d)
    c=d.Close
    r=c.shift(-h)/c-1
    allowed=x.iloc[:test_position-h].copy()
    allowed['target']=(r.iloc[:test_position-h]>0).astype(int)
    allowed=allowed.dropna()
    if len(allowed)<220 or allowed['target'].nunique()<2 or x.iloc[[-1]].isna().any(axis=None):
        return None
    model=_model(len(allowed));model.fit(allowed[FEATURES],allowed['target'])
    classes=list(model.classes_)
    pred=float(model.predict_proba(x.iloc[[-1]][FEATURES])[0,classes.index(1)])
    return pred,len(allowed),float(allowed["target"].mean())


@dataclass
class StrictAudit:
    recommendation: str
    explanation: str
    horizon_days: int
    evaluations: int
    evaluated_long_signals: int
    long_hit_rate: float | None
    long_wilson_lower: float | None
    long_avg_net_return: float | None
    buy_hold_rate: float | None
    overall_accuracy: float | None
    score_brier: float | None
    baseline_brier: float | None
    coverage: float | None
    positive_halves: bool
    rows: list[dict]
    diagnostics: list[str]
    similar_probability: float | None
    similar_count: int
    regime_reports: list[dict] = field(default_factory=list)
    reliability_bins: list[dict] = field(default_factory=list)


def audit_history(prices: pd.DataFrame, horizon_days:int, latest_score:float | None,
                  max_checks:int=36) -> StrictAudit:
    h=validate_horizon(horizon_days)
    frame=prices[['Open','High','Low','Close','Volume']].copy()
    n=len(frame)
    failures=[];rows=[]
    if n<650+h or latest_score is None:
        return StrictAudit('판단 보류','데이터 또는 학습 예측이 부족합니다.',h,0,0,None,None,None,None,None,None,None,None,False,[],['과거 차트 검증을 위한 기간이 부족합니다.'],None,0)
    # Test endpoints may not use future observations after each test date.
    # Non-overlapping realized outcomes, spaced no closer than H sessions.
    begin=max(420+h,int(n*.52))
    end=n-h-1
    if end<begin:
        return StrictAudit('판단 보류','과거 검증 표본 부족',h,0,0,None,None,None,None,None,None,None,None,False,[],['검증 가능한 미래 결과가 부족합니다.'],None,0)
    stride=max(h,max(1,(end-begin)//max_checks))
    positions=list(range(begin,end+1,stride))
    for t in positions:
        predicted=prediction_at(frame,t,h)
        if predicted is None:continue
        p,num_train,prior=predicted
        entry=float(frame.Close.iloc[t]);exit_=float(frame.Close.iloc[t+h]);actual_return=exit_/entry-1
        # Trend eligibility based exclusively on the date being evaluated.
        ma20=frame.Close.iloc[max(0,t-19):t+1].mean()
        ma60=frame.Close.iloc[max(0,t-59):t+1].mean()
        bull=(entry>ma60 and ma20>ma60)
        signal='매수 검토' if p>=BUY_THRESHOLD and bull else '매수 주의' if p<=AVOID_THRESHOLD else '관망'
        recent_vol=frame.Close.pct_change().iloc[max(0,t-19):t+1].std()
        historical_vol=frame.Close.pct_change().rolling(20).std().iloc[max(0,t-125):t+1].median()
        regime='고변동' if (pd.notna(recent_vol) and pd.notna(historical_vol) and recent_vol>historical_vol) else '저변동'
        rows.append({'날짜':frame.index[t].strftime('%Y-%m-%d') if hasattr(frame.index[t],'strftime') else str(frame.index[t]),
                     '모델상승점수':round(p,4),'신호':signal,
                     '실제수익률':round(actual_return,5),'실제상승':int(actual_return>0),
                     '정답':int((p>=0.5)==(actual_return>0)),'가상왕복비용차감수익률':round(actual_return-COST_FRACTION,5),
                     '훈련표본':num_train,'당시학습상승빈도':round(prior,4),'시장국면':regime,'상승추세':bull})
    m=len(rows)
    if m==0:
        return StrictAudit('판단 보류','독립된 과거 검증 결과가 없습니다.',h,0,0,None,None,None,None,None,None,None,None,False,[],['모델 학습 조건을 만족한 검증 구간이 없습니다.'],None,0)
    tab=pd.DataFrame(rows)
    mask=tab['신호']=='매수 검토'
    longs=tab.loc[mask]
    s=len(longs)
    successes=int((longs['가상왕복비용차감수익률']>0).sum())
    hit=successes/s if s else None
    lower=wilson_lower(successes,s)
    avg_net=float(longs['가상왕복비용차감수익률'].mean()) if s else None
    raw_acc=float(tab['정답'].mean())
    baseline=float(tab['실제상승'].mean())
    # At each test date, baseline is the past training-label prevalence, not the
    # retrospectively observed class mix or a convenient arbitrary 50% value.
    brier=float(brier_score_loss(tab['실제상승'],tab['모델상승점수']))
    base_brier=float(brier_score_loss(tab['실제상승'],tab['당시학습상승빈도']))
    # Do not cherry-pick successful dates or optimize thresholds on test outcomes.
    mid=m//2
    left=tab.iloc[:mid];right=tab.iloc[mid:]
    sides=[]
    for half in (left,right):
        long_half=half[half['신호']=='매수 검토']
        sides.append(len(long_half)>=3 and (long_half['가상왕복비용차감수익률']>0).mean()>=.55)
    consistent=all(sides)
    # A conservative filter uses sample size, fixed thresholds, relative results,
    # and a 95% Wilson lower bound. Passing is *not* a certified true probability.
    passes=(m>=MIN_TESTS and s>=MIN_SIGNALS and hit is not None
            and hit>=max(.62,baseline+.05) and lower is not None and lower>.50
            and avg_net is not None and avg_net>0 and consistent and brier<base_brier)
    recent=frame.Close
    now_bull=bool(recent.iloc[-1]>recent.tail(60).mean() and recent.tail(20).mean()>recent.tail(60).mean())
    if m<MIN_TESTS:failures.append(f'독립된 과거 평가 {m}건: 기준 {MIN_TESTS}건 미달')
    if s<MIN_SIGNALS:failures.append(f'과거 매수 검토 신호 {s}건: 기준 {MIN_SIGNALS}건 미달')
    if hit is not None and hit<max(.62,baseline+.05):failures.append('과거 매수 신호 적중률이 사전 고정된 통과 기준에 미달')
    if lower is not None and lower<=.50:failures.append('매수 신호 적중률의 95% 윌슨 하한이 50% 이하')
    if not consistent:failures.append('과거 검증기간의 전반·후반에서 모두 일관된 성적이 나타나지 않음')
    if brier>=base_brier:failures.append('전체 확률점수의 오차가 당시 과거 상승빈도 기준선보다 큼')
    if passes and latest_score>=BUY_THRESHOLD and now_bull:
        recommendation='조건부 매수 검토'
        explanation='과거 시간순 검증 필터를 통과했고 최신 상승점수·추세가 부합합니다. 미래 수익은 보장되지 않습니다.'
    else:
        recommendation='판단 보류'
        explanation='과거 검증을 통과하지 못했거나 현재 기술적 조건이 불일치합니다. 억지로 매수 신호를 만들지 않습니다.'
    grouped=[]
    for regime,grp in tab.groupby('시장국면',sort=False):
        buys=grp[grp['신호']=='매수 검토']
        grouped.append({'국면':regime,'독립 검증건수':int(len(grp)),
                        '매수 신호':int(len(buys)),
                        '비용 후 매수 적중률':float((buys['가상왕복비용차감수익률']>0).mean()) if len(buys)>=3 else None,
                        '전체 방향 적중률':float(grp['정답'].mean())})
    bins=[]
    for lower_bound,upper_bound in ((0.,.25),(.25,.4),(.4,.6),(.6,.75),(.75,1.00001)):
        group=tab[(tab['모델상승점수']>=lower_bound)&(tab['모델상승점수']<upper_bound)]
        bins.append({'점수구간':f'{lower_bound:.0%}~{min(upper_bound,1):.0%}',
                     '표본':int(len(group)),'평균 모델점수':float(group['모델상승점수'].mean()) if len(group)>=5 else None,
                     '실제 상승빈도':float(group['실제상승'].mean()) if len(group)>=5 else None})
    close_scores=tab[(tab['모델상승점수']-latest_score).abs()<=.1]
    similar_count=len(close_scores)
    similar=(float(close_scores['실제상승'].mean()) if similar_count>=10 else None)
    return StrictAudit(recommendation,explanation,h,m,s,hit,lower,avg_net,baseline,
                       raw_acc,brier,base_brier, s/m if m else None,consistent,
                       rows,failures,similar,similar_count,grouped,bins)
