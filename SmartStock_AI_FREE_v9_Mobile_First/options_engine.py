"""Equity options snapshot analytics (not order-flow direction or guaranteed support).

Uses available public option chains. OI ≠ directional positions; delayed snapshots
can't identify option writers, dealers or new trades. No network access in tests.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import date, datetime
import numpy as np
import pandas as pd

@dataclass
class OptionsReport:
    expiry: str
    dte: int
    call_oi: int
    put_oi: int
    call_volume: int
    put_volume: int
    oi_pcr: float | None
    volume_pcr: float | None
    max_pain: float | None
    call_wall: float | None
    put_wall: float | None
    atm_straddle: float | None
    atm_quote_kind: str
    implied_move_pct: float | None
    implied_vol: float | None
    otm_put_iv: float | None
    otm_call_iv: float | None
    skew_diff: float | None
    chain: pd.DataFrame
    call_greeks: dict | None
    put_greeks: dict | None
    notes: list[str]


def _num(x, default=math.nan):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _sanitize(src, typ):
    if src is None or not isinstance(src,pd.DataFrame) or src.empty:
        return pd.DataFrame(columns=['strike','openInterest','volume','impliedVolatility','bid','ask','lastPrice','type'])
    d=src.copy()
    for k in ['strike','openInterest','volume','impliedVolatility','bid','ask','lastPrice']:
        if k not in d:d[k]=np.nan
        d[k]=pd.to_numeric(d[k],errors='coerce')
    d=d[d['strike'].gt(0) & d['strike'].notna()].copy()
    d['openInterest']=d['openInterest'].fillna(0).clip(lower=0)
    d['volume']=d['volume'].fillna(0).clip(lower=0)
    d['type']=typ
    return d[['strike','openInterest','volume','impliedVolatility','bid','ask','lastPrice','type']]


def _mid(row):
    bid=_num(row.get('bid'))
    ask=_num(row.get('ask'))
    if math.isfinite(bid) and math.isfinite(ask) and ask>=bid and ask>0:
        return (bid+ask)/2, 'bid/ask 중간가'
    last=_num(row.get('lastPrice'))
    if math.isfinite(last) and last>0:return last,'최종 체결가(체결 시점 불명)'
    return None,'호가 없음'


def _cdf(x):return .5*(1+math.erf(x/math.sqrt(2)))
def _pdf(x):return math.exp(-x*x/2)/math.sqrt(2*math.pi)


def black_scholes_greeks(spot:float,strike:float,iv:float,dte:int,kind:str,rate:float=.04):
    """European model approximation; may deviate from American-style stock options."""
    if min(spot,strike,iv,dte)<=0:return None
    T=dte/365
    a=iv*math.sqrt(T)
    d1=(math.log(spot/strike)+(rate+iv*iv/2)*T)/a
    d2=d1-a
    delta=(_cdf(d1) if kind=='call' else _cdf(d1)-1)
    gamma=_pdf(d1)/(spot*a)
    vega=spot*_pdf(d1)*math.sqrt(T)/100 # for one vol percentage point
    theta_common=-(spot*_pdf(d1)*iv)/(2*math.sqrt(T))
    theta=(theta_common-rate*strike*math.exp(-rate*T)*_cdf(d2) if kind=='call'
           else theta_common+rate*strike*math.exp(-rate*T)*_cdf(-d2))/365
    return dict(delta=delta,gamma=gamma,vega_1pct=vega,theta_day=theta,iv=iv)


def analyze_options(spot:float,calls:pd.DataFrame,puts:pd.DataFrame,
                    expiry:str,reference_date:date|None=None,rate:float=.04)->OptionsReport:
    if spot<=0:raise ValueError('기초자산 가격이 유효하지 않습니다.')
    now=reference_date or datetime.now().date()
    dte=(date.fromisoformat(expiry)-now).days
    if dte<=0:raise ValueError('미래 만기 옵션을 선택해야 합니다.')
    c=_sanitize(calls,'CALL');p=_sanitize(puts,'PUT')
    if c.empty and p.empty:raise ValueError('유효한 옵션 체인이 없습니다.')
    co,po=int(c.openInterest.sum()),int(p.openInterest.sum())
    cv,pv=int(c.volume.sum()),int(p.volume.sum())
    pcr=po/co if co>0 else None
    vpcr=pv/cv if cv>0 else None
    chain=pd.concat([c,p],ignore_index=True).sort_values(['strike','type']).reset_index(drop=True)
    strikes=sorted(chain.strike.dropna().unique())
    max_pain=None
    if len(strikes)>0:
        # Intrinsic payout at expiration, weighted by reported open interest.
        # Contract multiplier cancels out only when all contracts use equal multiplier.
        pain=[(float(k),float(((k-c.strike).clip(lower=0)*c.openInterest).sum()+
                              ((p.strike-k).clip(lower=0)*p.openInterest).sum())) for k in strikes]
        max_pain=min(pain,key=lambda x:x[1])[0]
    call_wall=float(c.loc[c.openInterest.idxmax(),'strike']) if co and len(c) else None
    put_wall=float(p.loc[p.openInterest.idxmax(),'strike']) if po and len(p) else None
    c_atm=c.iloc[(c.strike-spot).abs().argsort()[:1]] if len(c) else c
    p_atm=p.iloc[(p.strike-spot).abs().argsort()[:1]] if len(p) else p
    c_row=c_atm.iloc[0] if len(c_atm) else None
    p_row=p_atm.iloc[0] if len(p_atm) else None
    cs,ct=_mid(c_row) if c_row is not None else (None,'호가 없음')
    ps,pt=_mid(p_row) if p_row is not None else (None,'호가 없음')
    straddle=cs+ps if cs is not None and ps is not None else None
    kind=ct if ct==pt else f'{ct} + {pt}'
    ivs=[_num(r.get('impliedVolatility')) for r in [c_row,p_row] if r is not None]
    ivs=[x for x in ivs if .001<x<10]
    atm_iv=float(np.mean(ivs)) if ivs else None
    em=(atm_iv*math.sqrt(dte/365)) if atm_iv is not None else None
    # OTM ~5% each side; not standardized risk reversals or delta-matched skew
    put_candidates=p[p.strike.lt(spot)]
    call_candidates=c[c.strike.gt(spot)]
    otm_put_iv=None;otm_call_iv=None
    if len(put_candidates):
        idx=(put_candidates.strike-spot*.95).abs().idxmin()
        x=_num(p.loc[idx,'impliedVolatility'])
        if .001<x<10:otm_put_iv=x
    if len(call_candidates):
        idx=(call_candidates.strike-spot*1.05).abs().idxmin()
        x=_num(c.loc[idx,'impliedVolatility'])
        if .001<x<10:otm_call_iv=x
    skew=otm_put_iv-otm_call_iv if otm_put_iv is not None and otm_call_iv is not None else None
    note=['풋/콜 OI 비율은 보유 계약의 수량 비율일 뿐 매수·매도 주체나 강세·약세를 확정하지 못합니다.',
          'Max Pain과 OI 집중 행사가가 실제 지지·저항 또는 종가 수렴을 보장하지 않습니다.',
          '옵션 체결가·미결제약정·내재변동성은 데이터 제공처에 따라 지연되거나 누락될 수 있습니다.',
          '1개월 예상 변동폭(IV)은 연율 내재변동성×√(잔존일수/365) 근사치로, 예측확률이나 보장범위가 아닙니다.']
    if straddle is not None and ct!='bid/ask 중간가' or straddle is not None and pt!='bid/ask 중간가':
        note.append('ATM 스트래들 값에 체결 시점 불명의 최종 체결가가 포함되어 정확도가 떨어질 수 있습니다.')
    cg=black_scholes_greeks(spot,float(c_row.strike),_num(c_row.impliedVolatility),dte,'call',rate) if c_row is not None else None
    pg=black_scholes_greeks(spot,float(p_row.strike),_num(p_row.impliedVolatility),dte,'put',rate) if p_row is not None else None
    note.append('그릭스는 유럽형 Black-Scholes 근사값(기본 금리 4%, 배당률 0%)이며, 실제 미국형 주식 옵션과 다를 수 있습니다.')
    return OptionsReport(expiry,dte,co,po,cv,pv,pcr,vpcr,max_pain,call_wall,put_wall,
                         straddle,kind,em,atm_iv,otm_put_iv,otm_call_iv,skew,chain,cg,pg,note)


def pick_expiry(expirations:list[str],today:date|None=None,target_dte:int=30):
    today=today or datetime.now().date()
    valid=[]
    for exp in expirations:
        try:
            d=(date.fromisoformat(exp)-today).days
            if d>0:valid.append((abs(d-target_dte),d,exp))
        except (ValueError,TypeError):pass
    return min(valid)[2] if valid else None
