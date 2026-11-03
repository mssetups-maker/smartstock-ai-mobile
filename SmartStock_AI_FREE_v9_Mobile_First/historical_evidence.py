"""Point-in-time-compatible historical evidence for SmartStock AI 11.2.

Two independent evidence sources are kept distinct:
1. Walk-forward random-forest predictions already audited in strict_validation.
2. A *fixed, non-trained* indicator voting rule replayed in chronological order.

The second is retrospective evidence of the stated rule, NOT independent proof
that the whole adaptive AI system forecasts future market prices.
"""
from __future__ import annotations
from datetime import timezone
import math
import numpy as np
import pandas as pd
from quality_guard import latest_signal

RULE_VERSION='fixed-indicators-v11.2.1'
MODEL_VERSION='strict-rf-v1'


def _iso_stamp(value) -> str:
    stamp=pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp=stamp.tz_localize('UTC')
    else:stamp=stamp.tz_convert('UTC')
    return stamp.isoformat()


def _bucket(score):
    try:
        if score is None or not math.isfinite(float(score)):return 'unavailable'
        v=float(score)
        return 'bull' if v>=.62 else 'bear' if v<=.38 else 'mixed'
    except (TypeError,ValueError):return 'unavailable'


def _outcome(return_value:float, threshold:float) -> str:
    return '상승' if return_value>threshold else '하락' if return_value<-threshold else '횡보'


def outcome_band(raw:pd.DataFrame, position:int, horizon:int) -> float|None:
    """Pre-registered market movement definition based on *past* volatility.

    The width is not a prediction probability. A class is defined as a forward
    absolute move exceeding max(0.5%, 0.30 * vol20 * sqrt(horizon)).
    """
    if position<21 or position+horizon>=len(raw):return None
    returns=raw.Close.iloc[:position+1].pct_change().tail(20)
    vol=float(returns.std())
    if not math.isfinite(vol):return None
    return max(.005, .30*vol*math.sqrt(horizon))


def fixed_rule_history(raw:pd.DataFrame,indicators_df:pd.DataFrame,horizon:int,
                       symbol:str,timeframe:str='1d',min_warmup:int=260) -> list[dict]:
    """Build non-overlapping, matured chronological evaluation events.

    The rule does not learn from the historical outcome, and each feature sees
    bars only through the event's date. The 0.5%/volatility label definition is
    held constant across dates; no post-hoc threshold fitting is performed.
    """
    if not isinstance(horizon,int) or horizon<1 or horizon>252:raise ValueError('기간 입력 오류')
    if len(raw)!=len(indicators_df) or not raw.index.equals(indicators_df.index):
        raise ValueError('지표 날짜와 시세 날짜가 다릅니다.')
    if not isinstance(raw.index,pd.DatetimeIndex) or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError('시계열 날짜 인덱스가 유효하지 않습니다.')
    n=len(raw)
    if n<min_warmup+2*horizon:return []
    # One complete H-bar window per observation, always strictly before now.
    # Sorting/spacing is independently verifiable, rather than an asserted flag.
    events=[];end_last=-1
    for i in range(min_warmup,n-horizon,horizon):
        if i<end_last:continue
        band=outcome_band(raw,i,horizon)
        if band is None:continue
        signal=latest_signal(indicators_df,row=i,
            volume_reliable=bool((raw.Volume.iloc[max(0,i-19):i+1]>0).sum()>=10))
        if signal['bucket']=='unavailable':continue
        px0=float(raw.Close.iloc[i]);px1=float(raw.Close.iloc[i+horizon]);
        if not (math.isfinite(px0) and math.isfinite(px1) and px0>0):continue
        realized=px1/px0-1
        t=_iso_stamp(raw.index[i]);end=_iso_stamp(raw.index[i+horizon])
        events.append({'event_id':f'{symbol}|{timeframe}|{horizon}|{t}|rule-v11.2',
                       'symbol':symbol,'timeframe':timeframe,'horizon_bars':horizon,
                       'signal_bucket':'rule:'+signal['bucket'],
                       'model_version':RULE_VERSION,'prediction_time':t,
                       'resolved_time':end,'actual_outcome':_outcome(realized,band),
                       'split':'out_of_sample','non_overlapping':True,
                       'entry':px0,'exit':px1,'realized_return':realized,
                       'neutral_band':band,
                       'evidence_source':'사전 고정 규칙의 과거 시간순 비중첩 재현 (비학습)'} )
        end_last=i+horizon
    return events


def model_history_from_audit(rows:list[dict],raw:pd.DataFrame,symbol:str,horizon:int,
                             timeframe:str='1d') -> list[dict]:
    """Re-use only previously walk-forward trained, matured strict OOS outputs.

    Reject rows with missing source dates or repeated / overlapping holding
    windows. Never infer model probabilities from latest in-sample fit.
    """
    if not isinstance(raw.index,pd.DatetimeIndex):return []
    pos_map={pd.Timestamp(d).date():i for i,d in enumerate(raw.index)}
    result=[];last_exit=-1
    for row in sorted(rows,key=lambda r:str(r.get('날짜',''))):
        try:
            date=pd.Timestamp(row['날짜']).date();i=pos_map[date]
            p=float(row['모델상승점수'])
            if i<last_exit or i+horizon>=len(raw) or not 0<=p<=1:continue
            r=float(raw.Close.iloc[i+horizon]/raw.Close.iloc[i]-1)
            # The audit's stored return must agree with actual bar prices.
            if not math.isclose(r,float(row['실제수익률']),abs_tol=2e-4):continue
            b=outcome_band(raw,i,horizon)
            if b is None:continue
            result.append({'event_id':f'{symbol}|{timeframe}|{horizon}|{_iso_stamp(raw.index[i])}|strict',
                       'symbol':symbol,'timeframe':timeframe,'horizon_bars':horizon,
                       'signal_bucket':'model:'+_bucket(p),'model_version':MODEL_VERSION,
                       'prediction_time':_iso_stamp(raw.index[i]),
                       'resolved_time':_iso_stamp(raw.index[i+horizon]),
                       'actual_outcome':_outcome(r,b),'split':'out_of_sample',
                       'non_overlapping':True,'score_at_prediction':p,
                       'neutral_band':b, 'evidence_source':'엄격 RF 시간순 재학습 독립 검증'})
            last_exit=i+horizon
        except (KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError):
            continue
    return result


def evidence_context(symbol:str,horizon:int,bucket:str,version:str,
                     asof, timeframe:str='1d') -> dict:
    return {'symbol':symbol,'timeframe':timeframe,'horizon_bars':horizon,
            'signal_bucket':bucket,'model_version':version,'asof':_iso_stamp(asof)}
