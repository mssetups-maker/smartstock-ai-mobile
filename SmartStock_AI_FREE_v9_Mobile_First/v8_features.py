"""Inspectable supplemental checks, not a prediction of guaranteed profit.

All calibration and screening scores are advisory; backtests are not a live trading result.
"""
from __future__ import annotations
from datetime import date
import math
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def reliability_calibration(rows, latest_raw_score=None):
    """Chronological calibration train/holdout. No use of holdout to fit calibrator.

    Evaluation itself is a single small OOS split and is therefore provisional.
    """
    if not rows or len(rows)<24:
        return {'status':'자료 부족: 보정 검증 최소 24건','train':0,'test':0,'calibrated_latest':None}
    frame=pd.DataFrame(rows).sort_values('날짜')
    x=pd.to_numeric(frame['모델상승점수'],errors='coerce').to_numpy(dtype=float)
    y=pd.to_numeric(frame['실제상승'],errors='coerce').to_numpy(dtype=float)
    mask=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=1)
    x=x[mask];y=y[mask]
    if len(x)<24 or np.unique(y).size<2:
        return {'status':'확률 보정 평가 표본 또는 상승/하락 사례 부족','train':0,'test':0,'calibrated_latest':None}
    n=max(12,int(len(x)*.6))
    if len(x)-n<10 or len(np.unique(y[:n]))<2:
        return {'status':'독립 후반부 검증 또는 전반부 클래스 부족','train':n,'test':len(x)-n,'calibrated_latest':None}
    model=IsotonicRegression(y_min=.0,y_max=1.,out_of_bounds='clip')
    model.fit(x[:n],y[:n])
    calibrated=np.clip(model.predict(x[n:]),0,1)
    orig=brier_score_loss(y[n:],x[n:])
    test=brier_score_loss(y[n:],calibrated)
    prior=brier_score_loss(y[n:],np.repeat(y[:n].mean(),len(y[n:])))
    passed=test<orig-.001 and test<prior-.001
    latest=None
    if passed and latest_raw_score is not None and math.isfinite(float(latest_raw_score)):
        # For the present day, all historical evaluated outcomes are already known.
        final=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(x,y)
        latest=float(final.predict([float(latest_raw_score)])[0])
    return {'status':('참고용 보정 가능(후반부 테스트 통과)' if passed else '보정 미채택: 후반부 성능 개선 없음'),
            'train':n,'test':len(x)-n,'raw_brier':float(orig),
            'cal_brier':float(test),'prior_brier':float(prior),
            'calibrated_latest':latest,'passed':passed}


def classify_market(prices):
    """Classify using only known historical closes, no hindsight switch."""
    close=prices.Close.dropna().astype(float)
    if len(close)<210:return {'regime':'자료 부족','warning':'시장 국면 분류에는 210개 이상 일봉이 필요합니다.'}
    ret=close.pct_change();vol20=float(ret.tail(20).std());
    median=float(ret.rolling(20).std().iloc[-210:-1].median())
    ma50=float(close.tail(50).mean());ma200=float(close.tail(200).mean());spot=float(close.iloc[-1]);
    highly_volatile=median>0 and vol20>1.6*median
    direction='상승' if spot>ma50>ma200 else '하락' if spot<ma50<ma200 else '횡보·혼조'
    regime=f'고변동 {direction}' if highly_volatile else direction
    return {'regime':regime,'vol_20':vol20,'vol_reference':median,'high_vol':highly_volatile,
            'ma50':ma50,'ma200':ma200,
            'warning':'고변동 구간입니다. 과거 평균 손절폭과 예측 성능이 달라질 수 있습니다.' if highly_volatile else ''}


def model_cost_stress(rows, friction_pct=.3):
    """Long-only price-close scenario; no broker execution, gaps or taxes modeled."""
    if not (0<=friction_pct<=10):raise ValueError('왕복 가정 비용은 0~10% 사이여야 합니다.')
    buys=[r for r in rows if r.get('신호')=='매수 검토']
    if not buys:return {'cases':0,'hit_rate':None,'avg_net':None,'worst':None}
    vals=np.array([float(r['실제수익률'])-friction_pct/100 for r in buys],dtype=float)
    vals=vals[np.isfinite(vals)]
    if not len(vals):return {'cases':0,'hit_rate':None,'avg_net':None,'worst':None}
    return {'cases':len(vals),'hit_rate':float(np.mean(vals>0)),
            'avg_net':float(np.mean(vals)),'worst':float(np.min(vals))}


def option_snapshot_changes(archive):
    if archive is None or archive.empty:return []
    d=archive.copy()
    vals=['pcr_oi','pcr_volume','atm_iv','iv_skew']
    if 'observed_day' not in d.columns:return []
    for col in vals:
        if col not in d:d[col]=np.nan
    d=d.sort_values('observed_day')
    output=[]
    for field,label in [('pcr_oi','풋/콜 OI 비율'),('pcr_volume','풋/콜 거래량 비율'),
                        ('atm_iv','ATM 내재변동성'),('iv_skew','풋·콜 IV 비대칭')]:
        observed=d[['observed_day',field]].dropna()
        if len(observed)<2:continue
        prev,last=observed.iloc[-2],observed.iloc[-1]
        output.append({'항목':label,'이전 실관측':float(prev[field]),'최근 실관측':float(last[field]),
                       '변화':float(last[field]-prev[field]),
                       '이전 관측일':str(pd.Timestamp(prev.observed_day).date()),
                       '최근 관측일':str(pd.Timestamp(last.observed_day).date())})
    return output


def earnings_event_info(calendar,now=None):
    """Provider calendar best-effort. No ungrounded earnings dates."""
    today=now or date.today()
    if calendar is None:return {'known':False,'days_to_event':None,'date':None}
    values=[]
    if isinstance(calendar,dict):
        for key,val in calendar.items():
            if 'earn' in str(key).lower() and 'date' in str(key).lower():
                values.extend(val if isinstance(val,(list,tuple)) else [val])
    elif isinstance(calendar,pd.DataFrame):
        for col in calendar.columns:
            if 'earn' in str(col).lower() and 'date' in str(col).lower():
                values.extend(calendar[col].dropna().to_list())
    for val in values:
        try:
            day=pd.Timestamp(val).date()
            if -1 <= (day-today).days <= 365:
                return {'known':True,'days_to_event':(day-today).days,'date':day.isoformat(),
                        'high_risk':0<=(day-today).days<=7}
        except (TypeError,ValueError,OverflowError):pass
    return {'known':False,'days_to_event':None,'date':None}


def indicator_candidate_lab(prices,horizon=21,friction_pct=.3):
    """Fixed candidate signals. Train-screen first period, blind-test latter period.

    This is *not* autonomous code generation; it is hypothesis screening.
    The most promising candidate is not auto-activated in the trading model.
    """
    from engine import indicators,validate_horizon
    h=validate_horizon(horizon)
    d=indicators(prices)
    if len(d)<(700+2*h):return {'status':'후보 탐색을 위한 기간 부족','rows':[]}
    rules={
        'MA20 > MA60': d['MA20']>d['MA60'],
        'MACD 히스토그램 양수': d['MACD_hist']>0,
        'RSI > 55':d['RSI14']>55,
        'Stoch 단기 K > D':d['STO_K5']>d['STO_D5'],
        '종가 > MA120':d['Close']>d['MA120'],
    }
    future=d.Close.shift(-h)/d.Close-1
    # Non-overlapping dates, training results known by the selection boundary.
    eval_indices=list(range(260,len(d)-h,h))
    cut=int(len(d)*.64)
    train=[i for i in eval_indices if i+h<cut]
    test=[i for i in eval_indices if i>=cut and i+h<len(d)]
    if len(train)<15 or len(test)<8:return {'status':'독립 검증 기간 부족','rows':[]}
    train_base=float(np.mean([future.iloc[i]>friction_pct/100 for i in train]))
    rows=[]
    for name,active in rules.items():
        a=[float(future.iloc[i])-friction_pct/100 for i in train if bool(active.iloc[i])]
        b=[float(future.iloc[i])-friction_pct/100 for i in test if bool(active.iloc[i])]
        if len(a)<7 or len(b)<5:continue
        train_hit=float(np.mean(np.array(a)>0));test_hit=float(np.mean(np.array(b)>0))
        test_prior=float(np.mean([future.iloc[i]>friction_pct/100 for i in test]))
        rows.append({'후보':name,'학습 신호 건수':len(a),'학습 적중률':train_hit,
                     '미사용 기간 신호 건수':len(b),'미사용 기간 적중률':test_hit,
                     '미사용 기간 평균 순수익':float(np.mean(b)),
                     '통과':(train_hit>train_base+.05 and test_hit>test_prior+.05
                            and float(np.mean(b))>0 and len(b)>=8)})
    return {'status':'실험 완료 (미사용 시기 통과 기준 보수적 적용)','rows':rows,
            'note':'임의로 만들어낸 지표가 아닌 고정 후보 5개를 평가합니다. 하나의 과거 미사용 기간을 재사용하므로 실전 적중을 보장하지 않습니다.'}


def watchlist_relative(frames):
    """Comparable snapshot metrics, without falsely representing ML ranking."""
    result=[]
    for symbol,df in frames.items():
        c=df.Close.astype(float).dropna();r=c.pct_change().dropna()
        if len(c)<65:continue
        result.append({'종목':symbol,'기준일':str(c.index[-1].date()),
                       '3개월 수익률':float(c.iloc[-1]/c.iloc[-64]-1),
                       '최근20일 변동성(연율)':float(r.tail(20).std()*np.sqrt(252)),
                       '60일 이평 대비':float(c.iloc[-1]/c.tail(60).mean()-1)})
    return sorted(result,key=lambda x:x['3개월 수익률'],reverse=True)


def interval_coverage_audit(prices,horizon=21,low_q=.10,high_q=.90):
    """Evaluate past empirical return bands on strict chronological test dates.

    Historical overlapping returns are *not* independent, so report holdout only
    at nonoverlapping test dates. This is an empirical coverage diagnostic, NOT a
    mathematically guaranteed predictive confidence interval.
    """
    from engine import validate_horizon
    h=validate_horizon(horizon)
    c=prices.Close.dropna().astype(float)
    if len(c)<max(650,6*h):return {'status':'검증 자료 부족','cases':0}
    returns=c.shift(-h)/c-1
    start=max(420,3*h)
    end=len(c)-h-1
    sample=[]
    for i in range(start,end,h):
        # At day i, realized returns with starting point <=i-h can be known.
        past=returns.iloc[:i-h+1].dropna()
        if len(past)<250:continue
        lo,hi=np.quantile(past,[low_q,high_q]);y=float(returns.iloc[i])
        if np.isfinite(y):sample.append((lo<=y<=hi,lo,hi,y))
    if len(sample)<12:return {'status':'독립 구간 부족','cases':len(sample)}
    past=returns.iloc[:-h].dropna()
    if len(past)<250:return {'status':'현시점 알려진 결과 부족','cases':len(sample)}
    lo,hi=np.quantile(past,[low_q,high_q]);now=float(c.iloc[-1])
    return {'status':'참고용 경험적 범위','cases':len(sample),
            'historical_coverage':float(np.mean([v[0] for v in sample])),
            'expected_coverage':high_q-low_q,'lower_price':now*(1+lo),'upper_price':now*(1+hi),
            'notes':'이동 경로가 아닌 선택 기간 후 종가 분포입니다. 과거 실현 수익 분포로만 계산하며 미래 보장 구간이 아닙니다.'}


def forecast_error_patterns(rows):
    """Descriptive error clustering only. Does not claim to identify causal reasons."""
    if not rows or len(rows)<12:return {'status':'오류 분석을 위한 독립 검증 사례 부족','cases':0}
    tab=pd.DataFrame(rows)
    x=pd.to_numeric(tab['모델상승점수'],errors='coerce')
    y=pd.to_numeric(tab['실제상승'],errors='coerce')
    wrong=((x>=.5)!=(y>0.5))
    examples=[]
    for name,mask in [('상승추세',tab['상승추세'].astype(bool)),
                      ('하락·횡보',~tab['상승추세'].astype(bool)),
                      ('고변동',tab['시장국면'].eq('고변동')),
                      ('저변동',tab['시장국면'].eq('저변동'))]:
        n=int(mask.sum())
        if n>=5:examples.append({'유형':name,'평가건수':n,'오분류건수':int((wrong&mask).sum()),
                                 '오분류율':float(wrong[mask].mean())})
    return {'status':'시장조건별 방향 예측 실패 패턴 (인과 원인 아님)',
            'cases':len(tab),'overall_error_rate':float(wrong.mean()),'groups':examples}
