"""Optional date-safe enhanced price + archived market-context model.

No trusted output without historical option + news observations. Full historical
options/news must have been archived at the time; current snapshots are not backfilled.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from engine import clean_prices,indicators,make_features,FEATURES,make_model,predict_prob,validate_horizon
from point_in_time import join_observations,FEATURE_NAMES,archive_stats

MIN_TRAIN=160
MIN_INDEPENDENT_TESTS=8
# Only stable features in first phase. Fundamentals archived and displayed, but
# sparse contemporaneous financial points are not silently imputed into the model.
CORE_EXTRA=['pcr_oi','atm_iv','news_sentiment']
OPTIONAL_FINANCIAL=['financial_revenue_growth','financial_profit_margin','financial_roe',
                    'sec_revenue_yoy','sec_roa']

@dataclass
class ContextAssessment:
    status:str
    explanation:str
    score:float|None
    holdout_cases:int
    brier_context:float|None
    brier_price_only:float|None
    brier_prior:float|None
    coverage:dict
    included_features:tuple[str,...]=()


def assess_context(prices:pd.DataFrame, archives:pd.DataFrame, horizon_days=21, sec_features=None)->ContextAssessment:
    """Evaluate price vs historical archive-enhanced model on held-out dates.

    NEWS, OPTIONS, and SEC are OPTIONAL groups. Historical public news must
    not be blocked because options history is unavailable. Features are chosen
    only from the training portion, not because they improved holdout results.
    """
    h=validate_horizon(horizon_days)
    prices=clean_prices(prices)
    extras=join_observations(prices.index,archives,max_age_days=7)
    coverage=archive_stats(extras)
    tech=make_features(indicators(prices))
    extra=extras.copy()
    if sec_features is not None and not sec_features.empty:
        for name in ('sec_revenue_yoy','sec_roa'):
            if name in sec_features:
                extra[name]=pd.to_numeric(sec_features[name].reindex(prices.index),errors='coerce')
    extra=extra.clip(-15,15)
    future=prices.Close.shift(-h)/prices.Close - 1
    labeled=pd.concat([tech,extra],axis=1)
    labeled=labeled.loc[future.notna()].copy()
    labeled['target']=(future.loc[future.notna()]>0).astype(int)
    labeled=labeled.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES)
    if len(labeled)<MIN_TRAIN+MIN_INDEPENDENT_TESTS*h:
        return ContextAssessment('비활성 — 기록 부족','기술지표의 과거 검증용 가격이 부족합니다.',None,0,None,None,None,coverage)
    # Pre-registered groups; do not cherry-pick indicators using future results.
    feature_groups=[('옵션',('pcr_oi','atm_iv')),
                    ('뉴스',('news_sentiment',)),
                    ('뉴스 건수',('news_count',)),
                    ('SEC 재무',('sec_revenue_yoy','sec_roa')),
                    ('관측 재무',('financial_revenue_growth','financial_profit_margin'))]
    cut=max(MIN_TRAIN,int(len(labeled)*.67))
    training_candidates=labeled.iloc[:max(0,cut-h)]
    selected=[];used_groups=[]
    for group, cols in feature_groups:
        if not all(c in labeled for c in cols):continue
        available=training_candidates[list(cols)].notna().all(axis=1)
        if len(training_candidates)>=MIN_TRAIN and int(available.sum())>=MIN_TRAIN and float(available.mean())>=.85:
            selected.extend(cols);used_groups.append(group)
    if not selected:
        return ContextAssessment('비활성 — 기록 부족',
            '과거에 확인 가능한 뉴스·옵션·공시 자료가 부족합니다. 무료 과거 뉴스 또는 SEC 공시를 먼저 검색할 수 있습니다.',
            None,0,None,None,None,coverage)
    # Evaluate without filling unavailable dates with invented values.
    clean=labeled.dropna(subset=selected)
    first_holdout_day=labeled.index[cut]
    train=clean[clean.index<first_holdout_day].iloc[:-h] if len(clean[clean.index<first_holdout_day])>h else clean.iloc[:0]
    # Keep independent evaluation at fixed real trading-day spacing.
    validation=labeled.iloc[cut::h].dropna(subset=selected)
    if len(train)<MIN_TRAIN-h or len(validation)<MIN_INDEPENDENT_TESTS or train.target.nunique()<2:
        return ContextAssessment('비활성 — 독립검증 부족',
            f'{"+".join(used_groups)} 자료는 있으나 {h}거래일 간격의 독립검증이 충분하지 않습니다.',
            None,len(validation),None,None,None,coverage,tuple(selected))
    xcols=FEATURES+selected
    m=make_model(len(train));m.fit(train[xcols],train.target)
    pctx=predict_prob(m,validation[xcols])
    baseline_model=make_model(len(train));baseline_model.fit(train[FEATURES],train.target)
    pprice=predict_prob(baseline_model,validation[FEATURES])
    prior=float(train.target.mean())
    a=validation.target.to_numpy()
    bc=float(brier_score_loss(a,pctx));bp=float(brier_score_loss(a,pprice))
    bn=float(brier_score_loss(a,np.full(len(a),prior)))
    current=pd.concat([tech,extra],axis=1).iloc[-1:]
    if current[xcols].isna().any(axis=None):
        return ContextAssessment('비활성 — 최신 자료 없음',
            '기록된 과거 자료는 있지만 현재 시점에 쓸 수 있는 뉴스·옵션·재무 자료가 없습니다.',
            None,len(a),bc,bp,bn,coverage,tuple(selected))
    if bc>=min(bp,bn):
        return ContextAssessment('검증 불통과 — 가격 모델 사용',
            '독립 검증에서 외부 자료 모델이 가격 전용 모델 및 당시 상승빈도 기준보다 확률오차가 낮지 않았습니다.',
            None,len(a),bc,bp,bn,coverage,tuple(selected))
    # Model selection based on validation can still overfit; require further
    # unseen prospective tests before treating this as a usable signal.
    final=make_model(len(clean));final.fit(clean[xcols],clean.target)
    score=float(predict_prob(final,current[xcols])[0])
    return ContextAssessment('검증 통과 — 참고 점수',
        f'{"+".join(used_groups)}+가격 보조모델입니다. 독립검증 일관성·종목 간 일반화 및 실전 적중률은 보장되지 않습니다.',
        score,len(a),bc,bp,bn,coverage,tuple(selected))
