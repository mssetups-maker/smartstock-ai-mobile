"""Free crypto OHLCV research and guarded stock/crypto transfer experiment.

Never labels candidate improvements 'verified probability' or auto-replaces the
live stock model. All stock evaluation samples are later in time than training
labels; crypto samples are restricted to the stock training cutoff.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, accuracy_score
from engine import FEATURES, indicators, make_features, clean_prices, validate_horizon

COINS = ('KRW-BTC', 'KRW-ETH', 'KRW-SOL')
UPBIT = 'https://api.upbit.com/v1/candles'


def get_candles(market: str, *, unit: str='days', count: int=1100,
                fetch: Callable | None = None, now: datetime | None = None) -> pd.DataFrame:
    """Download completed Upbit candles (not streaming). Supports day and 5m bars.

    A caller can supply fetch(url, params)->response for offline tests.
    """
    if market not in COINS:
        raise ValueError('지원 코인: KRW-BTC / KRW-ETH / KRW-SOL')
    if unit not in ('days', 'minutes/5') or not 1 <= count <= 4200:
        raise ValueError('unit은 days 또는 minutes/5, count는 1~4200')
    if fetch is None:
        import requests
        session = requests.Session()
        def fetch(url, params):
            reply = session.get(url, params=params, timeout=16,
                headers={'User-Agent':'SmartStockAI-Research/10.0'})
            reply.raise_for_status()
            return reply.json()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    interval = timedelta(days=1) if unit=='days' else timedelta(minutes=5)
    limit=min(count,4200)
    rows=[]; to=None
    while len(rows)<limit:
        params={'market':market,'count':min(200,limit-len(rows))}
        if to: params['to']=to
        batch=fetch(f'{UPBIT}/{unit}',params)
        if not isinstance(batch,list) or not batch:break
        rows.extend(batch)
        previous=pd.Timestamp(batch[-1]['candle_date_time_utc'],tz='UTC')
        to=previous.strftime('%Y-%m-%dT%H:%M:%SZ')
        if len(batch)<params['count']:break
        # Avoid hammering the public API on long backfill runs.
        if fetch is not None and len(rows)<limit: time.sleep(.14)
    frames=[]
    for x in rows:
        try:
            start=pd.Timestamp(x['candle_date_time_utc']).tz_localize('UTC')
            if start.to_pydatetime()+interval > now: continue  # unfinished candle
            frames.append((start.tz_localize(None),float(x['opening_price']),
                float(x['high_price']),float(x['low_price']),float(x['trade_price']),
                float(x['candle_acc_trade_volume'])))
        except (KeyError, TypeError, ValueError):continue
    df=pd.DataFrame(frames,columns=['Date','Open','High','Low','Close','Volume'])
    if df.empty:return pd.DataFrame(columns=['Open','High','Low','Close','Volume'], index=pd.DatetimeIndex([]))
    return clean_prices(df.set_index('Date'))


def training_rows(prices:pd.DataFrame, h:int) -> pd.DataFrame:
    """Known forward labels only: last H rows excluded, including missing labels."""
    h=validate_horizon(h)
    df=indicators(clean_prices(prices))
    X=make_features(df)
    forward=df.Close.shift(-h)/df.Close-1
    out=X.copy()
    out['target']=(forward>0).astype(int)
    out=out.loc[forward.notna()]
    return out.dropna()


def _train(X:pd.DataFrame, y:pd.Series, weight=None):
    model=RandomForestClassifier(n_estimators=90,max_depth=6,min_samples_leaf=7,
        random_state=120, n_jobs=1,class_weight='balanced_subsample')
    model.fit(X[FEATURES],y,sample_weight=weight)
    return model


def _proba(model, frame):
    return model.predict_proba(frame[FEATURES])[:,list(model.classes_).index(1)]


def transfer_audit(stock:pd.DataFrame, crypto:dict[str,pd.DataFrame], horizon:int=21) -> dict:
    """Never silently adopt; separate development and chronological test windows.

    Report effective (nonoverlapping) sample sizes, because daily targets share
    H-1 subsequent prices. Stocks and crypto have different market calendars.
    """
    h=validate_horizon(horizon)
    out={'horizon_days':h,'status':'검증 부족 — 기존 주식 모델 유지',
         'adopted':False,'reason':'독립적인 주식 검증이 완료되지 않았습니다.'}
    s=training_rows(stock,h)
    if len(s)<450:
        out['reason']=f'주식의 유효 학습 사례 {len(s)}개: 전이학습 비교에 필요한 시계열이 부족합니다.'
        return out
    # Final labeled stock bars contain no unobserved future outcomes.
    k=max(90,int(len(s)*.18))
    n=len(s)
    train_end=n-2*k-2*h
    dev_start=train_end+h
    test_start=dev_start+k+h
    if train_end<220 or test_start+k>n:
        out['reason']='기간별 시간순 학습/개발/독립검증 구간이 부족합니다.'
        return out
    train=s.iloc[:train_end];dev=s.iloc[dev_start:dev_start+k];test=s.iloc[test_start:test_start+k]
    # Crypto forward outcomes must have finished BEFORE the final stock training date.
    asof=pd.Timestamp(train.index[-1])
    pool=[];counts={};diagnostics={}
    # Crypto is traded every calendar day, unlike U.S. stocks. Match the
    # approximate *elapsed* horizon rather than using h calendar days.
    crypto_h=max(1,int(round(h*7/5)))
    for symbol,raw in crypto.items():
        try:
            cp=clean_prices(raw)
            observed_start=cp.index[0].isoformat() if len(cp) else None
            observed_end=cp.index[-1].isoformat() if len(cp) else None
            # Purge crypto bars whose outcomes could finish after the stock
            # training cutoff. Training_rows itself further excludes labels.
            cp=cp[cp.index<=asof-pd.Timedelta(days=crypto_h+1)]
            a=training_rows(cp,crypto_h) if len(cp)>140 else pd.DataFrame()
            eligible=len(a)
            # cap by evenly spaced observations; crypto must not swamp stocks
            if len(a)>len(train):
                idx=np.linspace(0,len(a)-1,len(train),dtype=int)
                a=a.iloc[idx]
            diagnostics[symbol]={'observed_start':observed_start,
                'observed_end':observed_end,'eligible_before_downsample':eligible,
                'eligible':len(a),'last_eligible':a.index[-1].isoformat() if len(a) else None}
            if len(a)>=40:
                pool.append(a);counts[symbol]=len(a)
        except Exception as exc:
            diagnostics[symbol]={'error':f'{type(exc).__name__}: {str(exc)[:100]}'}
    out.update({'stock_training_cases':len(train),'development_cases':len(dev),
        'holdout_cases':len(test),'effective_dev_cases':len(dev)//h,
        'effective_holdout_cases':len(test)//h,'crypto_training_cases':counts,
        'stock_training_cutoff':pd.Timestamp(train.index[-1]).isoformat(),
        'stock_test_start':pd.Timestamp(test.index[0]).isoformat(),
        'stock_test_end':pd.Timestamp(test.index[-1]).isoformat(),
        'crypto_horizon_calendar_days':crypto_h,
        'crypto_diagnostics':diagnostics})
    if not pool:
        out['reason']=f'주식 학습 종료일 {asof.date()}보다 충분히 오래된 코인 학습 사례가 없습니다. crypto_diagnostics를 확인하세요.'
        return out
    if train.target.nunique()<2:
        out['reason']='주식 학습 데이터에 상승·하락 양쪽 사례가 없습니다.'
        return out
    all_crypto=pd.concat(pool).sort_index()
    train2=pd.concat([train,all_crypto],ignore_index=True)
    a=_train(train[FEATURES],train.target)
    b=_train(train2[FEATURES],train2.target,
             weight=np.concatenate([np.ones(len(train)),np.full(len(all_crypto),.40)]))
    dp,cp=_proba(a,dev),_proba(b,dev)
    tp,up=_proba(a,test),_proba(b,test)
    metrics={
        'stock_dev_brier':round(float(brier_score_loss(dev.target,dp)),4),
        'stock_crypto_dev_brier':round(float(brier_score_loss(dev.target,cp)),4),
        'stock_holdout_brier':round(float(brier_score_loss(test.target,tp)),4),
        'stock_crypto_holdout_brier':round(float(brier_score_loss(test.target,up)),4),
        'stock_holdout_accuracy':round(float(accuracy_score(test.target,tp>=.5)),4),
        'stock_crypto_holdout_accuracy':round(float(accuracy_score(test.target,up>=.5)),4),
    }
    out.update(metrics)
    if len(dev)//h<12 or len(test)//h<12:
        out['reason']='겹치지 않는 2개 독립 평가 구간의 표본이 각각 12건에 미달합니다.'
        return out
    if metrics['stock_crypto_dev_brier'] >= metrics['stock_dev_brier'] or metrics['stock_crypto_holdout_brier'] >= metrics['stock_holdout_brier']:
        out['reason']='코인을 합친 모델이 두 시계열 평가 구간에서 일관된 개선을 입증하지 못했습니다.'
        return out
    out['status']='연구상 성능 개선 후보 — 실거래 모델 미교체'
    out['reason']='두 시간순 평가 구간에서 Brier가 개선됐습니다. 다른 종목·시장 국면의 반복 검증이 더 필요합니다.'
    return out
