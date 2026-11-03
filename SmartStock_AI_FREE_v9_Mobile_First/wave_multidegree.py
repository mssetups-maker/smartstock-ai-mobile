"""Multi-degree Elliott *structural hypotheses*, with historical counts preserved.

The patterns here are conservative geometrical screenings on confirmed pivots.  They
DO NOT verify each subwave's internal 5-3-5, 3-3-5, or 3-3-3-3-3 structure.
There is no objective unique Elliott labeling, no fitted probability, and no trade
advice.  As-of pivot confirmation always requires N candles to the right.

Conventions: impulse labels 0..5; an independent three-leg correction starts at
'시작' and ends A/B/C; triangle starts at '시작' and ends A..E; combinations show
small nested labels for their W/X/Y subpatterns.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from elliott_waves import _alternating_pivots, _impulse_valid, _live_invalidation

ORDERS = (3, 5, 8, 12, 21, 34)


def _axis(p: list[dict], d: int) -> list[float]:
    return [d * float(x['value']) for x in p]


def _direction(p: list[dict]) -> int:
    return 1 if p[1]['value'] > p[0]['value'] else -1


def _leg_floor(prices: pd.DataFrame) -> float:
    recent=prices.tail(min(len(prices),90))
    close=recent['Close'].to_numpy(dtype=float)
    high=recent['High'].to_numpy(dtype=float)
    low=recent['Low'].to_numpy(dtype=float)
    median=float(np.median(close))
    spreads=np.maximum(high-low,0)
    typical=float(np.median(spreads)) if len(spreads) else 0.0
    return max(1e-8,median*.0007,typical*.09)


def _valid_kinds(p: list[dict]) -> bool:
    return all(a['position']<b['position'] and a['kind'] != b['kind']
               for a,b in zip(p,p[1:]))


def _impulse(p: list[dict], floor: float) -> tuple[str,dict] | None:
    if not 3 <= len(p) <= 6 or not _valid_kinds(p):return None
    d=_direction(p)
    v=_axis(p,d)
    if v[1]-v[0] <= floor or not v[0]+floor*.1 < v[2] < v[1]-floor*.1:
        return None
    if len(p)==3:
        return ('충격파 초기 0-1-2',{'subtype':'초기형','hard_rules':['2파가 0파 시작을 침범하지 않음'],
                                   'internal_verified':False})
    if _impulse_valid(p,d,floor):
        return ('일반 충격파' if len(p)==6 else '일반 충격파 진행',
                {'subtype':'표준', 'hard_rules':['2파 시작점 미침범','3파 진행 방향','표준형 1·4 영역 비중첩'],
                 'internal_verified':False})
    if len(p)!=6:return None
    # At full 0..5 length consider a diagonal, not a mislabeled regular impulse.
    # For a diagonal, wave 4 can overlap wave 1, but 4 should not retrace 2.
    if not (v[3]>v[1]+floor*.1 and v[2]+floor*.1 < v[4] < v[1]):return None
    if not (v[5]>v[3]+floor*.1 and v[5]-v[4]>floor):return None
    ls=[v[1]-v[0],v[3]-v[2],v[5]-v[4]]
    if ls[1]+1e-8<min(ls[0],ls[2]):return None
    # Contracting / expanding diagonal only, not arbitrary overlapping five legs.
    contracting=ls[0]>ls[2] and (v[3]-v[1])>0 and (v[5]-v[3])>0
    expanding=ls[0]<ls[2] and (v[3]-v[1])>0 and (v[5]-v[3])>0
    if not (contracting or expanding):return None
    return ('대각형(다이아고날) 외형 후보',
            {'subtype':'수렴형' if contracting else '확산형',
             'hard_rules':['2파 시작점 미침범','4파 1파 영역 겹침','3파 최단 아님'],
             'internal_verified':False})


def _truncated_fifth(p: list[dict], floor:float, prices:pd.DataFrame, order:int) -> tuple[str,dict]|None:
    """Only promote an apparent truncated fifth after a smaller-scale 5 subwave.

    This deliberately errs on the side of not showing the rare subtype.  A mere
    failure to breach wave 3 never qualifies as a proven truncation.
    """
    if len(p)!=6 or not _valid_kinds(p):return None
    d=_direction(p); v=_axis(p,d)
    if not (v[0]<v[2]<v[1]<v[3] and v[1]<v[4]<v[3] and
            v[4]+floor < v[5] <= v[3]):return None
    if v[3]-v[2]<=floor:return None
    a,b=p[4]['position'],p[5]['position']
    # The smaller pivot subdivision must have six alternating points (0..5)
    # inside the bounds of 4-to-5, preserving motive direction.
    if b-a < 15:return None
    segment=prices.iloc[a:b+1]
    if len(segment)<18:return None
    for fine in (2,3):
        fine_p=_alternating_pivots(segment,fine,len(segment))
        # position in fine_p local; same direction and final motive bar near end.
        for j in range(max(0,len(fine_p)-18),len(fine_p)-5):
            q=fine_p[j:j+6]
            if _direction(q)==d and _impulse_valid(q,d,max(floor*.18,1e-8)):
                if q[0]['position']<=max(7,fine*3) and q[-1]['position']>=len(segment)-1-max(8,fine*4):
                    return ('5파 절단 후보 (하위 파동 관찰)',
                            {'subtype':'절단','hard_rules':['3파 고점 미돌파','하위 차수의 5개 진행파 존재'],
                             'internal_verified':False})
    return None


def _abc(p:list[dict], floor:float) -> tuple[str,dict]|None:
    """One 3-leg correction skeleton; type depends on retracement of B.

    Internal zigzag 5-3-5 and flat 3-3-5 are NOT verified. This is shape only.
    """
    if len(p)!=4 or not _valid_kinds(p):return None
    d=_direction(p);v=_axis(p,d)
    a=v[1]-v[0]
    if a<floor:return None
    b=(v[1]-v[2])/a
    if not .14<=b<=1.55:return None
    if v[3]-v[2]<floor:return None
    c=(v[3]-v[2])/a
    if not .25<=c<=2.5:return None
    if b<.85:
        if v[3]<=v[1]+floor*.1:return None  # zigzag C must pass A at same scale
        tag='지그재그 ABC 외형 후보'; rules=['B가 A 시작을 침범하지 않음','C가 A 끝을 넘어섬']
    elif b<=1.05:
        # A normal flat typically reaches near A extreme.
        if v[3]<v[1]-a*.2:return None
        tag='플랫 ABC 외형 후보';rules=['B가 A 시작점 부근까지 되돌림','C파 진행형']
    else:
        if v[3]>v[1]+floor*.1:
            tag='확장 플랫 ABC 외형 후보';rules=['B가 A 시작점을 넘어섬','C가 A 끝을 넘어섬']
        else:
            # Running flat extremely rare: do not label without internal proof.
            return None
    return (tag,{'subtype':tag.split(' ABC')[0], 'b_retrace':round(b,3),
                 'c_multiple':round(c,3),'hard_rules':rules,'internal_verified':False})


def _triangle(p:list[dict],floor:float) -> tuple[str,dict]|None:
    if len(p)!=6 or not _valid_kinds(p):return None
    # start, A, B, C, D, E. Contracting, completed triangle geometry.
    # A,C,E same pivot type, B,D alternate; swing range must contract.
    a,b,c,d,e=[float(z['value']) for z in p[1:]]
    if (p[1]['kind']=='L' and a+floor*.2<c and c+floor*.2<e and
        b>d+floor*.2 and b>c+floor and d>e+floor and b-a>d-c+floor*.1):
        kind='수렴 삼각형 ABCDE 외형 후보'
    elif (p[1]['kind']=='H' and a>c+floor*.2 and c>e+floor*.2 and
          b+floor*.2<d and c>b+floor and e>d+floor and a-b>c-d+floor*.1):
        kind='수렴 삼각형 ABCDE 외형 후보'
    elif (p[1]['kind']=='L' and c<a-floor*.2 and e<c-floor*.2 and
          d>b+floor*.2 and b>c+floor and d>e+floor):
        kind='확산 삼각형 ABCDE 외형 후보'
    elif (p[1]['kind']=='H' and c>a+floor*.2 and e>c+floor*.2 and
          d<b-floor*.2 and c>b+floor and e>d+floor):
        kind='확산 삼각형 ABCDE 외형 후보'
    else:return None
    subtype='수렴 삼각형' if '수렴' in kind else '확산 삼각형'
    return(kind,{'subtype':subtype,'hard_rules':['다섯 개의 교대 스윙','고저점의 기하학적 수렴 또는 확산'],
                 'internal_verified':False})


def _wxy(p:list[dict],floor:float) -> tuple[str,dict]|None:
    # W and Y each have an ABC-shaped subpattern, X is another 3-leg correction
    # pointing the other way. Explicitly avoid a false WXY label from only 3 pivots.
    if len(p)!=10 or not _valid_kinds(p):return None
    w,x,y=p[:4],p[3:7],p[6:10]
    cw,cx,cy=_abc(w,floor),_abc(x,floor),_abc(y,floor)
    if not(cw and cx and cy):return None
    if _direction(w)!=_direction(y) or _direction(w)==_direction(x):return None
    d=_direction(w)
    v=_axis(p,d)
    wmove=v[3]-v[0]
    if wmove<=floor:return None
    xretr=(v[3]-v[6])/wmove
    if not .18<=xretr<=1.1:return None
    if v[9]<=v[3]+floor*.1:return None
    return('복합 조정 W-X-Y 외형 후보',
           {'subtype':'WXY','hard_rules':['W/X/Y 각 3스윙 외형','X 반대 방향 되돌림','Y 재하락·재상승'],
            'x_retrace':round(xretr,3),'internal_verified':False})



def _wxyxz(p:list[dict],floor:float)->tuple[str,dict]|None:
    """Five adjacent ABC-like components: W-X-Y-X-Z (15 legs, 16 pivots).

    Only nested **geometrical** ABC skeletons are checked.  A triple correction
    is never labeled based merely on a 5-swing zigzag.
    """
    if len(p)!=16 or not _valid_kinds(p):return None
    groups=[p[k:k+4] for k in (0,3,6,9,12)]
    if not all(_abc(group,floor) for group in groups):return None
    d=_direction(groups[0])
    if [_direction(g) for g in groups] != [d,-d,d,-d,d]:return None
    v=_axis(p,d)
    w1=v[3]-v[0]; w2=v[9]-v[6]
    if min(w1,w2)<floor:return None
    x1=(v[3]-v[6])/w1
    x2=(v[9]-v[12])/w2
    if not (.18<=x1<=1.1 and .18<=x2<=1.1 and v[15]>v[9]+floor*.1):return None
    return('삼중 복합 조정 W-X-Y-X-Z 외형 후보',
           {'subtype':'WXYXZ','hard_rules':['W/X/Y/X/Z 각각 3스윙 외형',
             'X의 교차 되돌림 방향','Z가 Y 끝을 넘어섬'],
            'internal_verified':False})


def _make_nodes(p:list[dict],pattern:str)->list[dict]:
    if pattern.startswith('삼중 복합'):
        labels=['시작','w-a','w-b','W','x1-a','x1-b','X₁','y-a','y-b','Y','x2-a','x2-b','X₂','z-a','z-b','Z']
    elif pattern.startswith('복합'):
        labels=['시작','w-a','w-b','W','x-a','x-b','X','y-a','y-b','Y']
    elif '삼각형' in pattern:
        labels=['시작','A','B','C','D','E']
    elif 'ABC' in pattern:
        labels=['시작','A','B','C']
    else:
        labels=['0','1','2','3','4','5'][:len(p)]
    return [{'파동':label, '날짜':z['date'],'가격':round(float(z['value']),5),
             '종류':'고점' if z['kind']=='H' else '저점',
             'position':z['position'],'time':z['time'],'value':float(z['value'])}
            for z,label in zip(p,labels)]


def _result(p:list[dict], name:str, detail:dict, order:int, bars:int, floor:float)->dict:
    nodes=_make_nodes(p,name)
    end=nodes[-1]
    age=bars-1-int(p[-1]['position'])
    d=_direction(p)
    label=nodes[-1]['파동']
    if 'ABC' in name:
        current='ABC 외형 완료 · 내부 구조 및 상위 차수 검토 필요'
    elif '삼각형' in name:
        current='ABCDE 삼각형 외형 완료 · 돌파 방향 미확정'
    elif '복합' in name:
        current=('W-X-Y-X-Z 삼중 조정' if '삼중' in name else 'W-X-Y 복합 조정')+' 외형 완료 · 내부 구조 재검증 필요'
    elif len(p)==3:current='3파 진행 가능성 (0-1-2 후보)'
    elif len(p)==4:current='4파 조정 후보'
    elif len(p)==5:current='5파 진행 후보'
    else:current='5파 완료 가능 · 다음 구조 검토 필요'
    # A geometric screening score for comparative ordering, never a probability.
    duration=max(1,int(p[-1]['position'])-int(p[0]['position']))
    range_=abs(float(p[-1]['value'])-float(p[0]['value']))
    completeness=min(1,(len(p)-2)/5)
    scale_signal=min(1,range_/max(floor*18,1e-9))
    recency=math.exp(-max(0,age-order)/(max(12,order*5)))
    # Avoid claiming an objective goodness from overlapping heuristics.
    quality=round(30*completeness+25*scale_signal+35*recency+10*min(1,duration/(order*15)),2)
    if detail.get('subtype') in ('수렴형','확산형','절단'):quality=round(quality*.92,2)
    # context invalidation uses last confirmed pivot and geometry, not a stop order
    threshold=float(end['가격'])
    if '충격파' in name or '다이아고날' in name or '절단' in name:
        if label=='2':threshold=float(nodes[0]['가격']);inv='2파 0파 시작점 침범 시 무효'
        elif label=='3':threshold=float(nodes[1]['가격']);inv='표준 충격파의 4파가 1파 영역 침범 시 구조 재분류'
        elif label=='4':threshold=float(nodes[4]['가격']);inv='5파 진행 중 4파 끝 역행 시 재검토'
        else:threshold=float(nodes[-1]['가격']);inv='완료 이후 새 스윙 발생 시 다음 파동으로 별도 분석'
    else:
        threshold=float(nodes[-1]['가격']);inv='완료 후보는 다음 확정 스윙에서 상위 구조를 다시 판정'
    result={
        'status':'후보', 'direction':d, 'phase':name, 'current':current,
        'nodes':nodes,'anchors':[{'기준':'첫 기준점','날짜':nodes[0]['날짜'],'가격':nodes[0]['가격']},
            {'기준':'마지막 확정 스윙','날짜':end['날짜'],'가격':end['가격']}],
        'invalidation':{'price':round(threshold,5),'description':inv},
        'confirmed_through':end['날짜'], 'order':order,'pivot_count':len(p),
        'invalidated_candidates':0,'invalidation_event':None,
        'correction_status':name+' · 외형만 탐색',
        'projection':{'zone':None,'description':'하위 파동과 과거 예측 성능 검증 전 가격 목표를 생성하지 않습니다.'},
        'notes':(f'[{name}] {order}봉 민감도에서 {len(p)}개 기준점이 검출된 구조 **후보**입니다. '
                 '양쪽에 확인 봉이 있는 스윙만 번호를 붙이며 내부 하위 파동의 완전한 구조와 '
                 '유일한 파동 차수는 검증되지 않았습니다. 작도 점수는 실제 방향 확률이 아닙니다.'),
        'structure':name,'structure_subtype':detail.get('subtype'),
        'hard_rules':detail.get('hard_rules',[]),'internal_verified':detail.get('internal_verified',False),
        'heuristic_score':quality, 'start_position':p[0]['position'],'end_position':p[-1]['position'],
        'bars_since_last_confirmed':age,'lookback_bars':bars,
        'confirmation_bar_position':int(p[-1]['position'])+order,
        'degree_note':f'{order}봉 스윙에 기반한 차수 참고. 공식 Elliott 차수 확정 아님',
        'rank_method':'구조 길이·가격 변동폭·현재성·소요기간 휴리스틱 (미보정 점수)',
        'pattern_details':detail,
        'alternatives':[], 'history':[], 'requested_order':order,'auto_recounted':False,
    }
    return result


def _scan(prices:pd.DataFrame,order:int,lookback:int,max_pivots:int=90)->list[dict]:
    pivots=_alternating_pivots(prices,order,lookback)
    if not pivots:return []
    tail=pivots[-max_pivots:]
    # Determine price-scale floor using only candles available at each pivot.
    # Using today's price/ATR for a historical low-priced stock would wrongly
    # suppress old completed structures after splits or large appreciation.
    floors={z['position']:_leg_floor(prices.iloc[max(0,int(z['position'])-89):int(z['position'])+1])
            for z in tail}
    seen=set();out=[]
    lengths=(3,4,5,6,10,16)
    for length in lengths:
        for start in range(len(tail)-length+1):
            p=tail[start:start+length]
            if not _valid_kinds(p):continue
            floor=floors[p[-1]['position']]
            tests=[]
            if length<=6:
                v=_impulse(p,floor)
                if v: tests.append(v)
            if length==6:
                v=_truncated_fifth(p,floor,prices,order)
                if v:tests.append(v)
                v=_triangle(p,floor)
                if v:tests.append(v)
            if length==4:
                v=_abc(p,floor)
                if v:tests.append(v)
            if length==10:
                v=_wxy(p,floor)
                if v:tests.append(v)
            if length==16:
                v=_wxyxz(p,floor)
                if v:tests.append(v)
            for name,details in tests:
                # For incomplete impulse, later confirmed pivots can never be
                # presented as if this count is still active.
                key=(order,p[0]['position'],p[-1]['position'],name)
                if key in seen:continue
                seen.add(key)
                z=_result(p,name,details,order,len(prices),floor)
                z['has_later_confirmed_swing']=(start+length<len(tail))
                # Prospective invalidation checks for up to 5, with no false
                # invalidation of completed structures by a subsequent new leg.
                if name.startswith('일반') and 3<=len(p)<=5:
                    # original engine handles 0..3 and 0..4; 0..2 not supported.
                    if len(p)>=4:
                        breach=_live_invalidation(prices,p[-1],p,_direction(p))
                        if breach:
                            z['invalidation_event']=breach
                            z['invalidated_candidates']=1
                            continue
                    elif len(p)==3:
                        rest=prices.iloc[p[-1]['position']+1:]
                        if not rest.empty:
                            breach_price=float(rest['Low'].min() if _direction(p)==1 else rest['High'].max())
                            start_val=float(p[0]['value'])
                            if (_direction(p)==1 and breach_price<start_val) or (_direction(p)==-1 and breach_price>start_val):
                                continue
                out.append(z)
    return out


def _blank(order:int)->dict:
    return {'status':'판정 보류','direction':None,'phase':None,'current':'현재 유효 파동 없음',
            'nodes':[],'anchors':[],'invalidation':{'price':None,'description':'현재 작도를 강제하지 않습니다.'},
            'notes':'여러 차수·파형을 찾았으나 유효한 활성 후보가 없습니다. 과거 완료된 구조는 별도 보관합니다.',
            'confirmed_through':None,'order':order,'pivot_count':0,'invalidated_candidates':0,
            'invalidation_event':None,'alternatives':[],'history':[],
            'projection':{'zone':None,'description':'미산정'},'correction_status':'미확정',
            'recount_required':False,'auto_recounted':False,'requested_order':order,
            'recount_scales_tried':[],'previous_rejections':[], 'hard_rules':[],
            'internal_verified':False,'heuristic_score':None}


def analyze_wave_multidegree(prices:pd.DataFrame, requested_order:int=5,
                             lookback:int=1700,orders:tuple[int,...]|None=None)->dict:
    """Explore multiple scales; historical patterns do not disappear when stale.

    The primary 'active' selection must end at the last *confirmed* pivot in its
    scale.  Historical items are retrospectively labeled, NEVER prior forecasts.
    """
    if requested_order not in ORDERS:raise ValueError(f'허용 민감도: {ORDERS}')
    result=_blank(requested_order)
    if not isinstance(prices,pd.DataFrame) or len(prices)<45 or not {'Low','High','Close'}.issubset(prices.columns):
        return result
    if prices[['Low','High','Close']].isna().any().any():
        return result
    vals=prices[['Low','High','Close']].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals[:,0]>vals[:,1]).any() or (vals[:,0]>vals[:,2]).any() or (vals[:,2]>vals[:,1]).any():
        result['notes']='OHLC 자료의 수치 또는 고가/저가 범위 오류. 파동 표시를 차단했습니다.'
        return result
    scales=tuple(dict.fromkeys(orders or (requested_order,3,5,8,12,21,34)))
    scales=tuple(s for s in scales if s in ORDERS)
    if not scales:scales=(requested_order,)
    considered=[]
    for order in scales:
        considered.extend(_scan(prices,order,max(60,lookback)))
    result['recount_scales_tried']=list(scales)
    result['search_total']=len(considered)
    # Only one of the top candidates per kind/scale and date neighborhood.
    all_active=[]
    for z in considered:
        if z['has_later_confirmed_swing']:continue
        if z['bars_since_last_confirmed']>max(z['order']*4,32):continue
        all_active.append(z)
    active=sorted(all_active,key=lambda z:(
        z['heuristic_score']-abs(z['order']-requested_order)*.65,
        z['end_position'],len(z['nodes'])),reverse=True)
    diverse=[]
    for z in active:
        if any((z['structure']==old['structure'] and
                abs(z['end_position']-old['end_position'])<=max(2,z['order']) and
                abs(z['start_position']-old['start_position'])<=max(2,z['order']))
               for old in diverse):
            continue
        diverse.append(z)
        if len(diverse)>=4:break
    if diverse:
        result=dict(diverse[0])
        result['alternatives']=[dict(z,alternatives=[],history=[]) for z in diverse[1:]]
        result['auto_recounted']=result['order']!=requested_order
        result['recount_scales_tried']=list(scales)
        result['search_total']=len(considered)
        result['requested_order']=requested_order
        result['status']='후보'
    # Archive complete shapes even when current active is none.
    historic=[z for z in considered if len(z['nodes'])>=4 and
              ('충격파 진행' not in z['structure']) and
              ('초기 0-1-2' not in z['structure']) and
              (not z['structure'].startswith('일반') or len(z['nodes'])==6)]
    historic.sort(key=lambda z:(z['end_position'], z['heuristic_score']),reverse=True)
    archive=[]
    # Preserve at least one completed historical 0..5 impulse when available,
    # even if newer overlapping ABC skeletons would otherwise fill all slots.
    anchor_impulses=[z for z in historic if z['phase']=='일반 충격파' and len(z['nodes'])==6]
    for anchor in anchor_impulses:
        if result['nodes'] and not (anchor['end_position']<result['start_position'] or anchor['start_position']>result['end_position']):
            continue
        archive.append(dict(anchor,alternatives=[],history=[]))
        break
    for z in historic:
        # Non-overlapping structures, prefer the selected requested scale.
        if any(not (z['end_position'] < h['start_position'] or z['start_position'] > h['end_position'])
               for h in archive):continue
        if result['nodes'] and not (z['end_position']<result['start_position'] or z['start_position']>result['end_position']):
            continue
        archive.append(dict(z,alternatives=[],history=[]))
        if len(archive)>=4:break
    archive.sort(key=lambda z:z['start_position'])
    result['history']=archive
    result['historical_candidate_count']=len(historic)
    if not diverse:
        result['notes'] += (f' 탐색한 기하학적 구조 후보 {len(considered)}건 중 현재 활성 조건에 부합하는 후보는 0건입니다. '
                            '과거 완료 구조와 현재 파동의 구별은 유지합니다.')
    result['search_method']='서로 다른 확정 스윙 민감도에서 파형 외형을 탐색, 최근 활성 후보와 과거 완료 후보 분리'
    result['notes'] += (' 스윙 극점은 민감도에 해당하는 우측 봉이 나타난 후에만 확정됩니다. '
                        '기하학적 구조 검색과 하위 차수의 완전한 엘리엇 규칙 검증은 다릅니다.')
    return result


def wave_degree_consistency(prices:pd.DataFrame,orders=(3,5,8,12,21,34))->dict:
    rows=[]
    for k in orders:
        z=analyze_wave_multidegree(prices,requested_order=k,orders=(k,))
        rows.append({'민감도':k,'상태':z['status'],'구조':z.get('phase') or '—',
                     '진행 후보':z['current'],'마지막 기준일':z.get('confirmed_through') or '—'})
    keys={(r['상태'],r['구조'],r['진행 후보'],r['마지막 기준일']) for r in rows}
    return {'rows':rows,'agreement':len(keys)==1 and rows[0]['상태']=='후보',
            'notes':'차수별 파형 후보가 비슷한지만 확인합니다. 일치하더라도 미래 성공확률이 아닙니다.'}


def attach_multidegree_traces(fig:Any, chosen:dict, history:list[dict]|None=None,
                              cutoff:Any=None,latest:Any=None)->Any:
    """Active candidate plus at most four *historical hindsight* patterns."""
    import plotly.graph_objects as go
    def draw(result:dict,historical:bool):
        nodes=result.get('nodes',[])
        if cutoff is not None:
            nodes=[n for n in nodes if pd.Timestamp(n['time'])>=pd.Timestamp(cutoff)]
        if len(nodes)<2:return
        fig.add_trace(go.Scatter(x=[n['time'] for n in nodes],y=[n['가격'] for n in nodes],
             mode='lines+markers+text',text=[n['파동'] for n in nodes],
             textposition=['top center' if n['종류']=='고점' else 'bottom center' for n in nodes],
             textfont={'size':10 if historical else 12,'color':'#a4b9c6' if historical else '#ffffff'},
             line={'color':'#64748b' if historical else '#38c6ab',
                   'width':1.1 if historical else 2.2,
                   'dash':'dash' if historical else 'solid'},
             marker={'size':5 if historical else 9,'color':'#64748b' if historical else '#38c6ab'},
             name='과거 구조 외형(사후판정)' if historical else '현재 Elliott 구조 후보',
             showlegend=False,
             hovertemplate='%{text} · %{x}<br>가격 %{y:,.2f}<extra>' +
             ('과거 구조 사후판정' if historical else '현재 분석 후보')+'</extra>'),row=1,col=1)
    for h in (history or [])[:4]:draw(h,True)
    draw(chosen,False)
    nodes=chosen.get('nodes',[])
    if latest is not None and nodes:
        date,price=latest
        if pd.Timestamp(date)>pd.Timestamp(nodes[-1]['time']):
            fig.add_trace(go.Scatter(x=[nodes[-1]['time'],date],
               y=[nodes[-1]['가격'],float(price)],mode='lines+markers',
               line={'color':'#e2e8f0','width':1,'dash':'dot'},
               marker={'size':4},showlegend=False,name='마지막 미확정 구간'),row=1,col=1)
    return fig
