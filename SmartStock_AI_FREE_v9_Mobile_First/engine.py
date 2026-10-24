"""SmartStock AI — leakage-aware daily-bar feature engineering and research prototype.

The random forest is re-trained on available labeled data on each uncached analysis.
It does NOT autonomously invent indicators, provide certified probabilities or trade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

HORIZON = 21  # backwards-compatible default; use horizon_days for selected periods
MIN_HORIZON = 1
MAX_HORIZON = 252

def validate_horizon(horizon_days: int) -> int:
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, (int, np.integer)) or not MIN_HORIZON <= horizon_days <= MAX_HORIZON:
        raise ValueError(f"예측 기간은 {MIN_HORIZON}~{MAX_HORIZON}거래일의 정수여야 합니다.")
    return int(horizon_days)
FEATURES = [
    'ret5', 'ret21', 'ret63', 'ma20_dist', 'ma60_dist', 'ma120_dist',
    'ema_cross', 'rsi14_norm', 'macd_hist_norm', 'volatility20',
    'atr_pct', 'adx14_norm', 'stoch5_diff', 'stoch10_diff',
    'stoch20_diff', 'bb_position', 'volume_ratio', 'channel_position'
]
FEATURE_LABELS = {
    'ret5':'5거래일 모멘텀', 'ret21':'1개월 모멘텀', 'ret63':'3개월 모멘텀',
    'ma20_dist':'20일선과 이격', 'ma60_dist':'60일선과 이격',
    'ma120_dist':'120일선과 이격', 'ema_cross':'단기·장기 EMA 차이',
    'rsi14_norm':'RSI', 'macd_hist_norm':'MACD 히스토그램',
    'volatility20':'20일 변동성', 'atr_pct':'ATR 변동폭',
    'adx14_norm':'ADX 추세 강도', 'stoch5_diff':'단기 스토캐스틱',
    'stoch10_diff':'중기 스토캐스틱', 'stoch20_diff':'장기 스토캐스틱',
    'bb_position':'볼린저 밴드 위치', 'volume_ratio':'거래량 변화',
    'channel_position':'60일 가격 채널 위치',
}


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        raise ValueError('가격 데이터가 비어 있습니다.')
    df = prices.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    df.columns = [str(c).strip().title() for c in df.columns]
    expected = ['Open','High','Low','Close','Volume']
    missing = [k for k in expected if k not in df.columns]
    if missing:
        raise ValueError(f'필수 열이 없습니다: {", ".join(missing)}')
    df = df[expected].copy()
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Open','High','Low','Close'])
    df['Volume'] = df['Volume'].fillna(0).clip(lower=0)
    df = df[(df['Close'] > 0) & (df['Low'] > 0)]
    df = df[~df.index.duplicated(keep='last')].sort_index()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    return df


def wilder(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(alpha=1/window, adjust=False, min_periods=window).mean()


def stoch(df: pd.DataFrame, length: int, smooth: int, signal: int):
    ll = df['Low'].rolling(length).min()
    hh = df['High'].rolling(length).max()
    raw = 100 * (df['Close'] - ll) / (hh - ll).replace(0, np.nan)
    k = raw.rolling(smooth).mean()
    d = k.rolling(signal).mean()
    return k, d


def indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = clean_prices(data)
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    for n in [5, 10, 20, 60, 120, 240]:
        df[f'MA{n}'] = c.rolling(n).mean()
    df['EMA12'] = c.ewm(span=12, adjust=False).mean()
    df['EMA26'] = c.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    change = c.diff()
    up = wilder(change.clip(lower=0), 14)
    down = wilder((-change).clip(lower=0), 14)
    rs = up / down.replace(0, np.nan)
    df['RSI14'] = (100 - 100/(1+rs))
    df.loc[(down == 0) & (up > 0),'RSI14'] = 100.0
    df.loc[(up == 0) & (down > 0),'RSI14'] = 0.0
    df['RSI14'] = df['RSI14'].fillna(50)
    true_range = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df['ATR14'] = wilder(true_range, 14)
    plus_dm = h.diff()
    minus_dm = -l.diff()
    pos = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    neg = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    plus_di = 100 * wilder(pos, 14)/df['ATR14'].replace(0, np.nan)
    minus_di = 100 * wilder(neg, 14)/df['ATR14'].replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    df['ADX14'] = wilder(dx, 14)
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df['BB_upper'] = mid + 2*std
    df['BB_lower'] = mid - 2*std
    for n, sm, sig in [(5,3,3),(10,6,6),(20,12,12)]:
        df[f'STO_K{n}'], df[f'STO_D{n}'] = stoch(df, n, sm, sig)
    high9, low9 = h.rolling(9).max(), l.rolling(9).min()
    high26, low26 = h.rolling(26).max(), l.rolling(26).min()
    high52, low52 = h.rolling(52).max(), l.rolling(52).min()
    tenkan=(high9+low9)/2
    kijun=(high26+low26)/2
    df['ICH_A'] = ((tenkan+kijun)/2).shift(26)
    df['ICH_B'] = ((high52+low52)/2).shift(26)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['Close'].replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out['ret5'] = c.pct_change(5)
    out['ret21'] = c.pct_change(21)
    out['ret63'] = c.pct_change(63)
    out['ma20_dist'] = c / df['MA20'] - 1
    out['ma60_dist'] = c / df['MA60'] - 1
    out['ma120_dist'] = c / df['MA120'] - 1
    out['ema_cross'] = (df['EMA12'] - df['EMA26']) / c
    out['rsi14_norm'] = df['RSI14'] / 100
    out['macd_hist_norm'] = df['MACD_hist'] / c
    out['volatility20'] = c.pct_change().rolling(20).std()
    out['atr_pct'] = df['ATR14'] / c
    out['adx14_norm'] = df['ADX14'] / 100
    for n in [5,10,20]:
        out[f'stoch{n}_diff'] = (df[f'STO_K{n}'] - df[f'STO_D{n}']) / 100
    out['bb_position'] = (c - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']).replace(0,np.nan)
    out['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0,np.nan)
    rhi, rlo = df['High'].rolling(60).max(),df['Low'].rolling(60).min()
    out['channel_position'] = (c-rlo)/(rhi-rlo).replace(0,np.nan)
    return out[FEATURES].replace([np.inf, -np.inf], np.nan)


def make_model(n: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=170, max_depth=6, min_samples_leaf=max(6, n//250),
        max_features=0.72, class_weight='balanced_subsample',
        random_state=42, n_jobs=-1
    )


def predict_prob(m: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    cls = list(m.classes_)
    if 1 not in cls:
        return np.zeros(len(X))
    return m.predict_proba(X)[:,cls.index(1)]


def analog_paths(X: pd.DataFrame, df: pd.DataFrame, latest_idx: Any, n_neighbors: int = 45, horizon_days: int = HORIZON):
    """Past observations only, followed by the next selected horizon of actual bars.
    Candidates lacking a complete forward path are excluded.
    """
    horizon_days = validate_horizon(horizon_days)
    valid = X.dropna()
    if latest_idx not in valid.index:
        return None
    candidates = valid.iloc[:-horizon_days]
    if len(candidates) < 85:
        return None
    scaler = StandardScaler().fit(candidates[FEATURES].values)
    xs = scaler.transform(candidates[FEATURES].values)
    x0 = scaler.transform(valid.loc[[latest_idx],FEATURES].values)[0]
    distances = np.linalg.norm(np.clip(xs-x0, -5, 5), axis=1)
    selected = np.argsort(distances)[:min(n_neighbors,len(distances))]
    positions = df.index.get_indexer(candidates.index)
    close = df['Close'].values
    trajectories = []
    for ix in selected:
        i = positions[ix]
        if i >= 0 and i + horizon_days < len(df):
            trajectories.append(close[i:i+horizon_days+1] / close[i] - 1)
    if len(trajectories) < 10:
        return None
    a=np.array(trajectories)
    return dict(low=np.quantile(a,.10,axis=0),median=np.quantile(a,.50,axis=0),
                high=np.quantile(a,.90,axis=0), count=len(a))


def swing_levels(df:pd.DataFrame, order:int=4):
    """Centered pivots become known ONLY after `order` later bars."""
    lows=df['Low'].to_numpy()
    highs=df['High'].to_numpy()
    support, resistance=[],[]
    for i in range(order,len(df)-order):
        if lows[i] <= np.min(lows[i-order:i+order+1]):
            support.append((i,float(lows[i])))
        if highs[i] >= np.max(highs[i-order:i+order+1]):
            resistance.append((i,float(highs[i])))
    return support,resistance


def trade_levels(df:pd.DataFrame):
    close=float(df['Close'].iloc[-1]); atr=float(df['ATR14'].iloc[-1])
    if not math.isfinite(atr) or atr <= 0:
        atr=close*0.025
    supports,resists=swing_levels(df)
    support_below=[p for _,p in supports[-25:] if p < close]
    resistance_above=[p for _,p in resists[-25:] if p > close]
    support = max(support_below) if support_below else float(df['Low'].tail(45).min())
    resistance = min(resistance_above) if resistance_above else float(df['High'].tail(45).max())
    # Stops: recent swing is eligible only if reasonably close.
    sl = min(close - 1.6*atr, support - 0.2*atr) if close-support <= 3*atr else close - 1.8*atr
    sl = max(sl, close*.50)
    risk=max(close-sl,0.005*close)
    tp1=max(close+1.5*risk, resistance if resistance < close+3*risk else close+1.5*risk)
    tp2=close+2.5*risk
    return dict(entry=close,stop=sl,tp1=tp1,tp2=max(tp1+0.5*risk,tp2),
                support=support,resistance=resistance,atr=atr)


def classify(p:float, df:pd.DataFrame, enough_test:bool):
    last=df.iloc[-1]
    trend=(float(last['Close']) > float(last['MA60'])) and (float(last['MA20']) > float(last['MA60']))
    if not enough_test:
        return '판정 유보', '독립 검증 표본이 부족하므로 매수·매도 판단을 유보합니다.'
    if p >= 0.63 and trend:
        return '매수 검토', '모델 상승 추정치와 중기 추세가 함께 우호적입니다. 손절 기준을 먼저 확인하세요.'
    if p < 0.40 and not trend:
        return '매수 주의', '모델 상승 추정치와 중기 추세가 모두 약합니다.'
    return '관망', '모델 확률과 추세가 명확히 일치하지 않거나 확신이 낮습니다.'


@dataclass
class Analysis:
    probability: float | None
    backtest_accuracy: float | None
    backtest_brier: float | None
    baseline_accuracy: float | None
    test_cases: int
    train_cases: int
    model_features: list
    paths: dict | None
    levels:dict
    recommendation:str
    explanation:str
    diagnostics:list[str]


def analyze(df: pd.DataFrame, horizon_days: int = HORIZON) -> Analysis:
    horizon_days = validate_horizon(horizon_days)
    df=indicators(df) if 'ATR14' not in df.columns else df
    features=make_features(df)
    levels=trade_levels(df)
    latest=features.iloc[-1:]
    diagnostics=[]
    if len(df)<320:
        diagnostics.append('가격 이력이 320거래일 미만이므로 AI 학습을 실행하지 않았습니다.')
        return Analysis(None,None,None,None,0,0,[],None,levels,'판정 유보','충분한 데이터가 필요합니다.', diagnostics)
    if latest.isna().any(axis=1).iloc[0]:
        diagnostics.append('현재 지표 데이터가 부족하거나 결측되어 AI 확률을 계산하지 않았습니다.')
        return Analysis(None,None,None,None,0,0,[],None,levels,'판정 유보','지표 계산을 확인하세요.',diagnostics)
    future_r = df['Close'].shift(-horizon_days) / df['Close'] - 1
    labeled = features.copy()
    labeled['target'] = (future_r > 0).astype(int)
    # A future return of zero is valid; missing labels (last 21) MUST be excluded.
    labeled = labeled.loc[future_r.notna()].dropna()
    if len(labeled)<180 or labeled['target'].nunique()<2:
        diagnostics.append('AI 학습에 필요한 서로 다른 상승·하락 사례가 부족합니다.')
        return Analysis(None,None,None,None,0,0,[],None,levels,'판정 유보','학습 표본이 부족합니다.',diagnostics)
    cut=int(len(labeled)*0.72)
    if cut <= horizon_days+60:
        raise ValueError('학습 기간이 부족합니다.')
    # Purge: training outcomes may not reach into the first validation window.
    train=labeled.iloc[:cut-horizon_days]
    validation=labeled.iloc[cut:]
    model=make_model(len(train))
    model.fit(train[FEATURES],train['target'])
    # Select non-overlapping 21-session evaluation endpoints, not all overlapping bars.
    validation_ind=validation.iloc[::horizon_days]
    pval=predict_prob(model,validation_ind[FEATURES])
    actual=validation_ind['target'].to_numpy()
    count=len(actual)
    acc=float(accuracy_score(actual,pval>=0.5)) if count else None
    brier=float(brier_score_loss(actual,pval)) if count else None
    base_rate=float(train['target'].mean())
    baseline=float(accuracy_score(actual, np.full(count, int(train["target"].mean() >= 0.5)))) if count else None
    if count<12:
        diagnostics.append(f'{horizon_days}거래일 예측의 독립 검증 구간이 {count}건으로 적습니다(권장 12건 이상). 특히 장기 전망은 신뢰하기 어렵습니다.')
    if acc is not None and baseline is not None and acc < baseline:
        diagnostics.append('AI 백테스트 정확도가 단순 다수 클래스 기준보다 낮습니다. 결과를 매매 근거로 사용하지 마세요.')
    final=make_model(len(labeled))
    final.fit(labeled[FEATURES],labeled['target'])
    p=float(predict_prob(final,latest[FEATURES])[0])
    important=sorted(zip(FEATURES,final.feature_importances_),key=lambda z:z[1],reverse=True)[:6]
    paths=analog_paths(features,df,features.index[-1],horizon_days=horizon_days)
    recommendation, explanation=classify(p,df,count>=12 and acc is not None and acc>=.50 and acc>=baseline)
    return Analysis(p,acc,brier,baseline,count,len(labeled),important,paths,levels,recommendation,explanation,diagnostics)


def rule_summary(df: pd.DataFrame) -> list[tuple[str,str]]:
    last=df.iloc[-1]; c=float(last['Close']); x=[]
    for n in [20,60,120,240]:
        ma=float(last[f'MA{n}'])
        if math.isfinite(ma): x.append((f'{n}일 이동평균', '위' if c>=ma else '아래'))
    x.append(('RSI(14)', f"{float(last['RSI14']):.1f}"))
    x.append(('ADX(14)', f"{float(last['ADX14']):.1f}"))
    x.append(('MACD', '양(+) 모멘텀' if float(last['MACD_hist'])>0 else '음(-) 모멘텀'))
    return x
