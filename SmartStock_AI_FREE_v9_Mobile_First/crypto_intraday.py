"""Research-only 5-minute crypto data and one-hour out-of-sample evaluation.

This does NOT trade, publish verified probabilities, or modify stock models.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from engine import clean_prices

H=12  # 5 minute bars / 1 hour
COLUMNS=['ret1','ret3','ret12','ret48','vol12','vol48','ma12','ma48','range12','volume_ratio']


def test_one_hour(prices:pd.DataFrame)->dict:
    """Chronological purged test; evaluate outcomes no closer than 12 bars."""
    q=clean_prices(prices)
    out={'status':'검증 자료 부족','observations':len(q),'horizon_bars':H,
         'used_for_stock_prediction':False}
    if len(q)<620:
        out['reason']='5분봉 기록이 아직 620개 미만입니다.'
        return out
    c=q.Close
    f=pd.DataFrame(index=q.index)
    for k in (1,3,12,48):f[f'ret{k}']=c.pct_change(k)
    r=c.pct_change()
    for k in (12,48):f[f'vol{k}']=r.rolling(k).std()
    for k in (12,48):f[f'ma{k}']=c/c.rolling(k).mean()-1
    f['range12']=((q.High-q.Low)/c).rolling(12).mean()
    f['volume_ratio']=q.Volume/(q.Volume.rolling(48).mean().replace(0,np.nan))
    target_return=c.shift(-H)/c-1
    split=int(len(q)*.75)
    # Training labels known strictly before validation starts (purge 1h).
    train_ix=np.arange(0,split-H)
    val_ix=np.arange(split,len(q)-H,H)
    train=f.iloc[train_ix].copy();train['target']=(target_return.iloc[train_ix]>0).astype(int)
    train=train.replace([np.inf,-np.inf],np.nan).dropna()
    test=f.iloc[val_ix].replace([np.inf,-np.inf],np.nan).dropna()
    if len(train)<300 or len(test)<10 or train.target.nunique()<2:
        out['reason']=f'평가 가능한 훈련 {len(train)}건, 독립 검증 {len(test)}건으로 부족합니다.'
        return out
    y=(target_return.loc[test.index]>0).astype(int)
    model=RandomForestClassifier(n_estimators=85,max_depth=5,min_samples_leaf=8,random_state=17,n_jobs=1)
    model.fit(train[COLUMNS],train.target)
    p=model.predict_proba(test[COLUMNS])[:,list(model.classes_).index(1)]
    baseline=float(train.target.mean())
    brier=float(brier_score_loss(y,p))
    base=float(brier_score_loss(y,np.full(len(y),baseline)))
    out.update({'status':'연구 검증 완료 — 실거래 모델 미반영',
                'training_cases':len(train),'independent_test_cases':len(test),
                'model_brier':round(brier,5),'historical_baseline_brier':round(base,5),
                'model_improved_on_test':bool(brier<base),
                'test_period_start':str(test.index[0]),'test_period_end':str(test.index[-1]),
                'reason':'하나의 과거 검증구간 결과로 미래 정확도나 수익을 보장하지 않습니다.'})
    return out
