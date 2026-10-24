"""Point-in-time price-data integrity and transparent risk calculations.

Estimates are descriptive, never calibrated predictions of future tail losses.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, date
import math
import numpy as np
import pandas as pd

@dataclass
class DataAudit:
    status: str
    warnings: list[str]
    age_weekdays: int | None
    rows: int
    last_date: str
    missing_volume_pct: float

@dataclass
class RiskReport:
    ann_volatility: float | None
    max_drawdown: float | None
    daily_var95: float | None
    daily_cvar95: float | None
    worst_daily_return: float | None
    worst_open_gap: float | None
    median_dollar_turnover_20d: float | None
    last_20d_atr_pct: float | None
    entry_price: float
    stop_price: float
    tp1_price: float
    stop_distance_pct: float
    reward_risk: float | None
    sample_days: int


def data_audit(prices:pd.DataFrame, as_of:date|None=None)->DataAudit:
    if as_of is None:
        as_of=datetime.now(timezone.utc).date()
    msgs=[]
    if prices is None or prices.empty:
        return DataAudit('차단',['가격 데이터가 비어 있습니다.'],None,0,'—',100.)
    c=prices['Close'].astype(float)
    o=prices['Open'].astype(float)
    h=prices['High'].astype(float)
    l=prices['Low'].astype(float)
    v=prices['Volume'].astype(float)
    last=pd.Timestamp(prices.index[-1]).date()
    age=int(np.busday_count(np.datetime64(last),np.datetime64(as_of))) if last<=as_of else -1
    missing_volume=float((v.tail(min(60,len(v)))<=0).mean())
    if age<0:
        msgs.append('시세 기준일이 현재 날짜보다 미래입니다. 거래소 시간·CSV 날짜를 확인하세요.')
    elif age>=5:
        msgs.append(f'마지막 확정 일봉이 평일 기준 {age}일 이전입니다. 오래된 가격으로 현재 매수 판단을 하지 않습니다.')
    elif age>=3:
        msgs.append(f'마지막 확정 일봉이 평일 기준 {age}일 이전입니다. 최신 가격을 확인하세요.')
    if len(prices)<700:
        msgs.append(f'일봉 {len(prices)}개: 장기 예측용 과거 검증 구간이 충분하지 않을 수 있습니다.')
    if missing_volume>=.5:
        msgs.append(f'최근 거래량 0 또는 누락 비율 {missing_volume:.0%}: 거래량 기반 지표를 신뢰하기 어렵습니다.')
    if ((h < np.maximum(o,c)*.9999) | (l > np.minimum(o,c)*1.0001) | (l>h)).any():
        msgs.append('고가·저가·시가·종가 관계에 오류가 있는 행이 있습니다. CSV나 원시 데이터를 확인하세요.')
    daily=c.pct_change().replace([np.inf,-np.inf],np.nan).dropna()
    if len(daily) and (daily.abs()>0.70).any():
        msgs.append('하루 ±70% 이상 가격 변동이 관찰됩니다. 분할·데이터 오류 또는 극단적 변동을 점검하세요.')
    # Old and impossible dates are hard blockers. Volume problems are cautionary.
    hard=age<0 or age>=5 or ((h<np.maximum(o,c)*.9999)|(l>np.minimum(o,c)*1.0001)|(l>h)).any()
    status='차단' if hard else '주의' if msgs else '정상'
    return DataAudit(status,msgs,age,len(prices),last.isoformat(),missing_volume)


def risk_report(prices:pd.DataFrame, levels:dict, lookback:int=252)->RiskReport:
    p=prices.tail(max(lookback+1,70)).copy()
    c=p.Close.astype(float)
    simple=c.pct_change().replace([np.inf,-np.inf],np.nan).dropna()
    ret=simple.tail(lookback)
    enough=len(ret)>=40
    vol=float(ret.std()*np.sqrt(252)) if enough else None
    var=float(ret.quantile(.05)) if enough else None
    cvar=float(ret[ret<=var].mean()) if enough and var is not None and (ret<=var).any() else None
    worst=float(ret.min()) if enough else None
    rolling_peak=c.cummax()
    dd=(c/rolling_peak-1).dropna()
    max_dd=float(dd.min()) if len(dd) else None
    gap=(p.Open.astype(float)/p.Close.shift().astype(float)-1).replace([np.inf,-np.inf],np.nan).dropna()
    worst_gap=float(gap.min()) if len(gap)>40 else None
    vol20=(prices.Close.tail(20)*prices.Volume.tail(20)).replace([np.inf,-np.inf],np.nan)
    med_turnover=float(vol20.median()) if len(vol20)>=12 else None
    # The risk budget assumes fills at the selected entry/stop; gaps may exceed it.
    entry=float(levels['entry'])
    stop=float(levels['stop'])
    target=float(levels['tp1'])
    risk=max(entry-stop,0)
    rrr=float((target-entry)/risk) if risk>0 else None
    return RiskReport(vol,max_dd,var,cvar,worst,worst_gap,med_turnover,
             float(levels['atr']/entry) if entry>0 else None,entry,stop,target,
             float(risk/entry) if entry>0 else 0,rrr,len(ret))


def hypothetical_size(entry:float,stop:float,portfolio:float,risk_percent:float,
                      max_allocation_percent:float=20.0)->dict:
    """Illustrative whole-share position sizing by maximum planned stop-loss cost.
    Returns zero for invalid setups. Gaps and fees may produce larger losses.
    """
    if not all(math.isfinite(float(x)) for x in (entry,stop,portfolio,risk_percent,max_allocation_percent)):
        raise ValueError('숫자를 입력하세요.')
    if entry<=0 or portfolio<=0 or stop<=0 or stop>=entry:
        raise ValueError('진입가·총자산이 양수이고 손절가가 진입가보다 낮아야 합니다.')
    if not 0 < risk_percent <= 10 or not 0 < max_allocation_percent<=100:
        raise ValueError('위험 비중 0~10%, 최대 배분 0~100% 범위에서 설정하세요.')
    budget=portfolio*risk_percent/100
    loss_per_share=entry-stop
    by_stop=int(budget//loss_per_share)
    by_exposure=int((portfolio*max_allocation_percent/100)//entry)
    qty=max(0,min(by_stop,by_exposure))
    return {'shares':qty,'position_value':qty*entry,'planned_loss_to_stop':qty*loss_per_share,
            'risk_budget':budget,'planned_allocation_pct':qty*entry/portfolio*100,
            'budget_limited_by':'손절 위험' if by_stop<by_exposure else '최대 비중'}


def historical_stress(prices:pd.DataFrame, periods:tuple[int,...]=(5,21,63))->pd.DataFrame:
    """Historical worst / median / best H-bar outcomes (overlapping; NOT out-of-sample)."""
    c=prices.Close.astype(float)
    rows=[]
    for h in periods:
        r=(c/c.shift(h)-1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(r)<70:continue
        rows.append({'기간(거래일)':h,'최악 과거 수익률':float(r.min()),
                     '하위 5%':float(r.quantile(.05)),
                     '중앙값':float(r.median()),'상위 95%':float(r.quantile(.95)),
                     '표본(중첩)':len(r)})
    return pd.DataFrame(rows)

def compare_market(stock:pd.DataFrame,benchmark:pd.DataFrame,periods:tuple[int,...]=(21,63,126,252))->tuple[dict,pd.DataFrame,pd.DataFrame]:
    """Aligned close-to-close comparisons; no ML training or lookahead."""
    s=stock.Close.rename('종목')
    b=benchmark.Close.rename('비교지수')
    both=pd.concat([s,b],axis=1,join='inner').dropna().sort_index()
    if len(both)<65:raise ValueError('동일한 거래일로 비교할 수 있는 데이터가 65일 미만입니다.')
    returns=both.pct_change().dropna().tail(252)
    var=float(returns['비교지수'].var())
    beta=float(returns['종목'].cov(returns['비교지수'])/var) if var>0 else None
    corr=float(returns['종목'].corr(returns['비교지수'])) if len(returns)>30 else None
    rows=[]
    for h in periods:
        if len(both)>h:
            stock_gain=float(both['종목'].iloc[-1]/both['종목'].iloc[-h-1]-1)
            bench_gain=float(both['비교지수'].iloc[-1]/both['비교지수'].iloc[-h-1]-1)
            rows.append({'기간(거래일)':h,'종목':stock_gain,'비교지수':bench_gain,'초과 수익률':stock_gain-bench_gain})
    normalized=both.tail(252)/both.tail(252).iloc[0]*100
    return {'beta':beta,'correlation':corr,'aligned_rows':len(both),
            'last_stock_date':str(stock.index[-1].date()),'last_benchmark_date':str(benchmark.index[-1].date())},pd.DataFrame(rows),normalized
