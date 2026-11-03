"""Causal OHLCV checks and transparent, grouped indicator diagnostics.

Rule agreement is *not* a calibrated return probability. Correlated indicators
are pooled inside trend / momentum / volume groups before the final ranking.
No trained model is allowed to override independent historical evidence gates.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from engine import indicators
from market_data_guard import check_ohlcv

PRICE_FEATURES = ('MA20','MA60','MA120','MA240','EMA12','EMA26','MACD','MACD_signal',
                  'MACD_hist','ATR14','ADX14','RSI14','BB_upper','BB_lower',
                  'STO_K5','STO_D5','STO_K10','STO_D10','STO_K20','STO_D20',
                  'MFI14','OBV','OBV_slope20')
RANGED = ('RSI14','ADX14','MFI14','STO_K5','STO_D5','STO_K10','STO_D10','STO_K20','STO_D20')
GROUPS = ('추세','모멘텀','수급')
DIRECTION = {1:'상승',-1:'하락',0:'중립'}
RULE_ID = 'grouped-consensus-v11.3'


def verify_and_rebuild(raw:pd.DataFrame, supplied:pd.DataFrame|None=None)->dict[str,Any]:
    """Recompute from OHLCV, compare supplied indicator values and reject corrupt bars."""
    check_ohlcv(raw)
    if not isinstance(raw.index,pd.DatetimeIndex):
        raise ValueError('가격 날짜 인덱스를 확인하세요.')
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError('중복·역순 가격 날짜를 정리한 뒤 분석해야 합니다.')
    reference=indicators(raw)
    if len(reference)<90:raise ValueError('검증 가능한 지표 기간이 부족합니다.')
    last=reference.iloc[-1]
    blocked=[];skipped=[];repaired=[]
    for col in PRICE_FEATURES:
        if col not in reference:
            blocked.append(f'{col}: 계산 항목 누락')
            continue
        v=last[col]
        if pd.isna(v):
            skipped.append(f'{col}: 현 시점 계산 자료 부족/가격 변화 또는 거래량 부족')
            continue
        if not math.isfinite(float(v)):
            blocked.append(f'{col}: 비정상 숫자')
            continue
        if col in RANGED and not -1e-6<=float(v)<=100+1e-6:
            blocked.append(f'{col}: 0~100 범위 위반')
        if col=='ATR14' and float(v)<0:blocked.append('ATR14: 음수')
        if supplied is not None:
            if col not in supplied or supplied.empty or pd.isna(supplied[col].iloc[-1]):
                repaired.append(f'{col}: 재산정')
                continue
            older=float(supplied[col].iloc[-1]);fresh=float(v)
            if not math.isfinite(older) or not math.isclose(older,fresh,rel_tol=1e-5,abs_tol=1e-7):
                repaired.append(f'{col}: 이전 값 불일치로 재산정')
    macd=float(last['MACD']);sig=float(last['MACD_signal']);hist=float(last['MACD_hist'])
    if not math.isclose(hist,macd-sig,rel_tol=1e-7,abs_tol=1e-8):
        blocked.append('MACD 히스토그램 일관성 오류')
    if last['BB_upper'] < last['BB_lower']:
        blocked.append('볼린저 상하단 관계 오류')
    vol=raw['Volume'].tail(20)
    volume_reliable=bool((vol>0).sum()>=10)
    if not volume_reliable:
        skipped.append('최근 거래량 불충분: MFI·OBV 판단에서 제외')
    if blocked:
        return {'status':'차단','df':reference,'issues':blocked,'skipped':skipped,'repaired':repaired,
                'volume_reliable':volume_reliable}
    return {'status':'재산정' if repaired else '정상','df':reference,'issues':[],
            'skipped':skipped,'repaired':repaired,'volume_reliable':volume_reliable}


def _finite(value:Any)->float|None:
    try:
        if value is None or isinstance(value,bool):return None
        number=float(value)
        return number if math.isfinite(number) else None
    except (ValueError,TypeError,OverflowError):return None


def _vote(value:Any,low:float,high:float)->int|None:
    v=_finite(value)
    return None if v is None else 1 if v>high else -1 if v<low else 0


def _fmt(value:Any,precision:int=2)->str:
    n=_finite(value)
    return '자료 없음' if n is None else f'{n:,.{precision}f}'


def _snapshot(frame:pd.DataFrame,row:int=-1):
    """Latest bar position, with no opportunity to inspect later bars."""
    pos=row if row>=0 else len(frame)+row
    if pos<0 or pos>=len(frame):raise IndexError('분석 봉 위치가 유효하지 않습니다.')
    return frame.iloc[pos],pos


def latest_signal(frame:pd.DataFrame,volume_reliable:bool=True,row:int=-1)->dict[str,Any]:
    """Fixed causal rules with inspectable measurements and equal-weight groups.

    Within a group, votes are averaged. Between the available groups, group
    means are averaged; 3 moving-average rules therefore do not act as three
    independent market observations. `bucket` is intentionally coarse for
    chronological fixed-rule history, not a learned directional probability.
    """
    if frame.empty:
        return {'bucket':'unavailable','bias':0.,'votes':[],'available':0,
                'groups':[],'direction':'판단 보류','conflicts':0,'rule_id':RULE_ID}
    s,pos=_snapshot(frame,row)
    close=_finite(s.get('Close'))
    if close is None or close<=0:
        return {'bucket':'unavailable','bias':0.,'votes':[],'available':0,
                'groups':[],'direction':'판단 보류','conflicts':0,'rule_id':RULE_ID}
    votes=[]
    date=str(frame.index[pos].date()) if isinstance(frame.index,pd.DatetimeIndex) else str(frame.index[pos])

    def add(label,group,value,lo,hi,display,criterion):
        direction=_vote(value,lo,hi)
        if direction is not None:
            votes.append({'indicator':label,'group':group,'vote':direction,'measured':value,
                          'value':display,'rule':criterion,'asof':date})

    ma60=_finite(s.get('MA60')); ma20=_finite(s.get('MA20'));ma120=_finite(s.get('MA120'))
    if ma60 is not None and ma60>0:
        relative=close/ma60-1
        add('60일 이동평균 추세','추세',relative,-.005,.005,
            f'종가 {_fmt(close)} / MA60 {_fmt(ma60)} ({relative:+.2%})','MA60 이격 ±0.5%')
    if ma20 is not None and ma60 is not None and ma60>0:
        relative=ma20/ma60-1
        add('20/60일 이동평균 배열','추세',relative,-.003,.003,
            f'MA20 {_fmt(ma20)} / MA60 {_fmt(ma60)} ({relative:+.2%})','배열 차이 ±0.3%')
    if ma60 is not None and ma120 is not None and ma120>0:
        relative=ma60/ma120-1
        add('60/120일 이동평균 배열','추세',relative,-.003,.003,
            f'MA60 {_fmt(ma60)} / MA120 {_fmt(ma120)} ({relative:+.2%})','배열 차이 ±0.3%')
    rsi=_finite(s.get('RSI14'))
    add('RSI(14)','모멘텀',rsi,45,55,_fmt(rsi),'45 미만 하락 / 55 초과 상승')
    hist=_finite(s.get('MACD_hist'))
    if hist is not None:
        normalized=hist/close
        add('MACD 히스토그램','모멘텀',normalized,-.0003,.0003,
            f'{_fmt(hist,4)} (종가 대비 {normalized:+.3%})','종가 대비 ±0.03%')
    stoch_k=_finite(s.get('STO_K20'));stoch_d=_finite(s.get('STO_D20'))
    if stoch_k is not None and stoch_d is not None:
        add('슬로우 스토캐스틱(20)','모멘텀',stoch_k-stoch_d,-4,4,
            f'%K {_fmt(stoch_k)} / %D {_fmt(stoch_d)} (차이 {stoch_k-stoch_d:+.2f})','K-D ±4')
    if volume_reliable:
        mfi=_finite(s.get('MFI14'))
        add('MFI(14)','수급',mfi,45,55,_fmt(mfi),'45 미만 하락 / 55 초과 상승')
        obv=_finite(s.get('OBV_slope20'))
        add('OBV 거래량 추세','수급',obv,-1e-10,1e-10,_fmt(obv,0)+'주 (20봉 변화)','20봉 변화 부호 (0은 중립)')
    group_rows=[]
    for group in GROUPS:
        subset=[v for v in votes if v['group']==group]
        if subset:
            mean=sum(v['vote'] for v in subset)/len(subset)
            group_rows.append({'group':group,'mean':mean,'up':sum(v['vote']>0 for v in subset),
                               'down':sum(v['vote']<0 for v in subset),
                               'neutral':sum(v['vote']==0 for v in subset),
                               'count':len(subset)})
    g=len(group_rows)
    bias=sum(group['mean'] for group in group_rows)/g if g else 0.
    # Three groups (or two when reliable volume is unavailable) is the maximum
    # independent vote count. A bullish + bearish conflict is not certainty.
    bucket='bull' if len(votes)>=4 and bias>=.25 else 'bear' if len(votes)>=4 and bias<=-.25 else 'mixed' if len(votes)>=4 else 'unavailable'
    direction={'bull':'상승 우세','bear':'하락 우세','mixed':'혼조/박스 가능성','unavailable':'판단 보류'}[bucket]
    return {'bucket':bucket,'bias':bias,'votes':votes,'available':len(votes),
            'groups':group_rows,'independent_groups':g,'direction':direction,
            'conflicts':sum(v['vote']!=0 and v['vote']*bias<0 for v in votes),
            'rule_id':RULE_ID,'asof':date}


def regime_diagnostic(frame:pd.DataFrame,row:int=-1)->dict[str,Any]:
    """Separate a short pullback from longer-term trend evidence (never guarantee reversal)."""
    if frame.empty:return {'label':'판단 보류','reason':'가격 자료가 없습니다.','status':'unavailable'}
    s,pos=_snapshot(frame,row)
    ma20=_finite(s.get('MA20'));ma60=_finite(s.get('MA60'));ma120=_finite(s.get('MA120'))
    px=_finite(s.get('Close'));rsi=_finite(s.get('RSI14'))
    if px is None or any(x is None for x in (ma20,ma60,ma120)) or pos<20:
        return {'label':'중장기 구조 검증 불충분','reason':'이동평균 20·60·120일 자료가 부족합니다.','status':'unavailable'}
    previous120=_finite(frame['MA120'].iloc[pos-20])
    if previous120 is None:
        return {'label':'중장기 구조 검증 불충분','reason':'20봉 전 MA120 자료가 부족합니다.','status':'unavailable'}
    slope=ma120/previous120-1 if previous120>0 else 0.
    long_bull=ma60>ma120 and slope>0 and px>=ma120
    long_bear=ma60<ma120 and slope<0 and px<ma120
    short_weak=px<ma20 and ma20<ma60 and (rsi is None or rsi<50)
    short_strong=px>ma20 and ma20>ma60 and (rsi is None or rsi>50)
    note=f'종가 {_fmt(px)}, MA20 {_fmt(ma20)}, MA60 {_fmt(ma60)}, MA120 {_fmt(ma120)}; MA120의 20봉 변화 {slope:+.2%}'
    if long_bull and short_weak:
        return {'label':'장기 상승 구조 내 단기 조정 후보','reason':note+' · 장기 배열은 상승이지만 단기선·모멘텀이 약합니다. 추세 전환 확정이 아닙니다.','status':'pullback'}
    if long_bear and short_strong:
        return {'label':'장기 하락 구조 내 단기 반등 후보','reason':note+' · 단기 반등과 장기 추세 회복은 다릅니다.','status':'rebound'}
    if long_bull:
        return {'label':'중장기 상승 구조 유지','reason':note+' · 구조가 유지되지만 향후 상승을 보장하지 않습니다.','status':'bullish'}
    if long_bear:
        crosses=frame.iloc[max(1,pos-40):pos+1]
        prev_diff=(crosses['MA60']-crosses['MA120']).shift(1)
        recent_cross=bool(((prev_diff>=0)&((crosses['MA60']-crosses['MA120'])<0)).fillna(False).any())
        last_two_below=bool((frame.Close.iloc[max(0,pos-1):pos+1]<frame.MA120.iloc[max(0,pos-1):pos+1]).all())
        if recent_cross and last_two_below:
            return {'label':'최근 중장기 하락 전환 조건 충족 후보','reason':note+' · 최근 40봉 내 MA60의 MA120 하향 교차와 2봉 종가 아래를 확인했지만 거짓 신호 가능성이 있습니다.','status':'possible_reversal'}
        return {'label':'중장기 하락 구조','reason':note+' · 언제 시작된 추세인지는 별도 확인이 필요합니다.','status':'bearish'}
    return {'label':'중장기 혼조·전환 미확정','reason':note+' · 장기 배열과 현재 가격·기울기가 충분히 일치하지 않습니다.','status':'mixed'}


def _price_levels(frame:pd.DataFrame|None)->dict[str,Any]:
    """Return conditional boundaries computed only from already-observed candles.

    A threshold is a *rule reevaluation marker*, NOT an empirically calibrated
    stop, an order suggestion, nor a guaranteed reversal price.
    """
    if frame is None or len(frame)<30:return {}
    s=frame.iloc[-1]
    atr=_finite(s.get('ATR14'));close=_finite(s.get('Close'))
    ma60=_finite(s.get('MA60'))
    # Freeze the reference range BEFORE the two most recent completed bars
    # so a two-close invalidation can actually be evaluated without hindsight.
    lo=_finite(frame['Low'].iloc[-22:-2].min())
    hi=_finite(frame['High'].iloc[-22:-2].max())
    if any(v is None for v in (atr,close,ma60,lo,hi)) or atr<=0 or close<=0 or hi<=lo:
        return {}
    buffer=.10*atr
    return {'prior20_low':lo,'prior20_high':hi,'atr':atr,'close':close,'ma60':ma60,
            'up_break':max(hi,ma60)+buffer,'down_break':min(lo,ma60)-buffer,
            'range_low':lo,'range_high':hi,'asof':str(frame.index[-1].date())}


def scenario_candidates(signal:dict[str,Any],model_direction_score:float|None=None,
                        frame:pd.DataFrame|None=None)->list[dict]:
    """Three scenarios ranked by equally-weighted group agreement, not probability.

    The RF score is deliberately NOT counted as an extra group because it was
    trained on the same technical inputs. The displayed group scores are ordinal
    rule-consensus coverage; no % probability is inferred from them.
    """
    if signal.get('available',0)<4 or not signal.get('groups'):return []
    group_rows=signal['groups'];n=len(group_rows)
    support={
        '상승':sum(max(0.,g['mean']) for g in group_rows)/n,
        '하락':sum(max(0.,-g['mean']) for g in group_rows)/n,
        '횡보':sum(1-abs(g['mean']) for g in group_rows)/n,
    }
    # Their sum is one because they partition group AGREEMENT, not future outcomes.
    levels=_price_levels(frame)
    model_note=('별도 미보정 RF 상승점수 '+f'{float(model_direction_score):.3f}'+' (중복 입력이므로 순위 가중치 0)'
        if _finite(model_direction_score) is not None and 0<=float(model_direction_score)<=1 else 'RF 점수 없음')
    invalidations={
        '상승':'가격 구조 기준이 부족하여 수치 무효화 미제시',
        '하락':'가격 구조 기준이 부족하여 수치 무효화 미제시',
        '횡보':'가격 구조 기준이 부족하여 수치 무효화 미제시',
    }
    if levels:
        invalidations={
            '상승':f'완성된 일봉 종가가 {levels["down_break"]:,.2f} 밑에서 2봉 연속 마감하면 상승 해석 재검토. (평가 직전 2개 봉 이전의 20봉 저점/MA60·0.1ATR 완충)',
            '하락':f'완성된 일봉 종가가 {levels["up_break"]:,.2f} 위에서 2봉 연속 마감하면 하락 해석 재검토. (평가 직전 2개 봉 이전의 20봉 고점/MA60·0.1ATR 완충)',
            '횡보':f'평가 직전 2개 봉 이전의 20봉 범위 {levels["range_low"]:,.2f}~{levels["range_high"]:,.2f} 밖에서 2봉 연속 종가 마감하면 횡보 해석 재검토.',
        }
    # Invalidate a directional *structure* only after two confirmed closes
    # beyond the reference built before those two closes. A conditional future
    # rebound is still possible, but that rejected structure is never displayed.
    invalidated={}
    if levels and frame is not None:
        last_two=frame['Close'].iloc[-2:]
        seen=f'{float(last_two.iloc[0]):,.2f}, {float(last_two.iloc[1]):,.2f}'
        if bool((last_two<levels['down_break']).all()):
            invalidated['상승']=f'최근 완성 일봉 2개 종가 {seen}가 상승 구조 재검토선 {levels["down_break"]:,.2f} 아래'
        if bool((last_two>levels['up_break']).all()):
            invalidated['하락']=f'최근 완성 일봉 2개 종가 {seen}가 하락 구조 재검토선 {levels["up_break"]:,.2f} 위'
        if bool((last_two>levels['range_high']).all()) or bool((last_two<levels['range_low']).all()):
            invalidated['횡보']=f'최근 완성 일봉 2개 종가 {seen}가 기존 횡보범위 {levels["range_low"]:,.2f}~{levels["range_high"]:,.2f} 밖'
    results=[]
    for outcome,slug in (('상승','up'),('하락','down'),('횡보','side')):
        component=[]
        for g in group_rows:
            contribution=max(0,g['mean']) if outcome=='상승' else max(0,-g['mean']) if outcome=='하락' else 1-abs(g['mean'])
            component.append(f'{g["group"]} 기여 {100*contribution/n:.1f}점 / {100/n:.1f}점 (그룹 내부 평균 {g["mean"]:+.2f})')
        direction_support=[v['indicator']+' '+v['value'] for v in signal['votes'] if
                           (v['vote']>0 if outcome=='상승' else v['vote']<0 if outcome=='하락' else v['vote']==0)]
        evidence=component+[f'점수 계산: {100*support[outcome]:.1f}점 = 3계열 동일 비중의 규칙 일치도 (확률 아님)',
                            f'{model_note}']
        if direction_support:evidence.append('일치하는 원시 지표: '+ ' / '.join(direction_support[:4]))
        else:evidence.append('현재 해당 방향의 직접 지표 지지가 없으며 조건부 대안에 불과합니다.')
        # Mixed (side) is not an actual sideways forecast. Its rule score
        # indicates abstention/disagreement in the available groups.
        if outcome=='횡보':evidence.append('횡보 점수는 혼조·중립 비중이며 실제 향후 박스권 확률이 아닙니다.')
        results.append({'id':'rule-'+slug,'title':{'상승':'상승 방향 조건부 시나리오',
                       '하락':'하락 방향 조건부 시나리오','횡보':'혼조·횡보 관찰 시나리오'}[outcome],
                        'valid':outcome not in invalidated,'data_valid':True,'outcome':outcome,
                        'reject_reason':invalidated.get(outcome,'해당 없음'),
                        'rank_score':round(100*support[outcome],4),
                        'evidence':evidence[:8], 'invalidation':invalidations[outcome],
                        'origin':RULE_ID})
    return results


def indicator_table(signal:dict[str,Any])->pd.DataFrame:
    return pd.DataFrame([{'계열':v.get('group','미분류'),'지표':v['indicator'],
             '실측값':v.get('value','자료 없음'),
             '상태':DIRECTION.get(v['vote'],'미확인'),
             '판정 기준':v.get('rule','-'),
             '기준일':v.get('asof','-')}
             for v in signal.get('votes',[])],
             columns=['계열','지표','실측값','상태','판정 기준','기준일'])


def group_table(signal:dict[str,Any])->pd.DataFrame:
    return pd.DataFrame([{'계열':g['group'],'상승':g['up'],'하락':g['down'],
                          '중립':g['neutral'],'지표 수':g['count'],
                          '계열 내부 방향':_fmt(g['mean']),
                          '합의 공식':f'({g["up"]}-{g["down"]})/{g["count"]}'}
                         for g in signal.get('groups',[])],
                        columns=['계열','상승','하락','중립','지표 수','계열 내부 방향','합의 공식'])
