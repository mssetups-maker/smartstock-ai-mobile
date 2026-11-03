"""Conservative Elliott impulse/ABC hypotheses on *confirmed* pivots only.

This is a transparent pattern heuristic, not a unique Elliott count, forecast,
trading signal, or trained model.  Pivot i is only confirmed after `order`
subsequent bars.  The last live leg is therefore ALWAYS a hypothesis.
"""
from __future__ import annotations

from typing import Any
import math
import pandas as pd
from engine import swing_levels


def _alternating_pivots(prices: pd.DataFrame, order: int, lookback: int) -> list[dict[str, Any]]:
    x = prices.tail(lookback)
    offset = len(prices) - len(x)
    lows, highs = swing_levels(x, order=order)
    events = []
    def readable_time(t):
        stamp=pd.Timestamp(t)
        return stamp.strftime('%Y-%m-%d') if (stamp.hour,stamp.minute,stamp.second)==(0,0,0) else stamp.strftime('%Y-%m-%d %H:%M')
    for n, price in lows:
        events.append({'position': offset+n, 'date': readable_time(prices.index[offset+n]),
                       'time': prices.index[offset+n], 'value': float(price), 'kind': 'L'})
    for n, price in highs:
        events.append({'position': offset+n, 'date': readable_time(prices.index[offset+n]),
                       'time': prices.index[offset+n], 'value': float(price), 'kind': 'H'})
    events.sort(key=lambda r: r['position'])
    selected: list[dict[str, Any]] = []
    conflicted_positions = {p['position'] for p in events if sum(e['position'] == p['position'] for e in events)>1}
    for row in events:
        if row['position'] in conflicted_positions:
            continue   # same-bar inside/outside candle is not an unambiguous turning-point sequence
        if selected and selected[-1]['kind'] == row['kind']:
            if (row['kind']=='H' and row['value'] > selected[-1]['value']) or (row['kind']=='L' and row['value'] < selected[-1]['value']):
                selected[-1] = row
        else:
            selected.append(row)
    return selected


def _impulse_valid(p: list[dict], direction: int, min_leg: float) -> bool:
    """Validate 0..up-to-5 confirmed pivots for a *standard* impulse only.

    Bullish coordinates rise when price rises, bearish coordinates likewise via
    multiplication by -1.  Wave 4 must not overlap wave 1 for normal impulses.
    A diagonal / truncated wave 5 is intentionally not recognized.
    """
    if len(p) < 4 or len(p) > 6:
        return False
    kinds = [('L' if i%2==0 else 'H') if direction==1 else
             ('H' if i%2==0 else 'L') for i in range(len(p))]
    if [a['kind'] for a in p] != kinds:
        return False
    v = [direction * x['value'] for x in p]
    if v[1]-v[0] < min_leg or not (v[0] < v[2] < v[1]):
        return False
    if v[3]-v[2] < min_leg or v[3] <= v[1]:
        return False
    if len(v) >= 5 and not (v[1] < v[4] < v[3]):
        return False
    if len(v) == 6:
        # Extended third wave, fifth wave must exceed third; third cannot be shortest.
        if v[5] <= v[3] or v[5]-v[4] < min_leg:
            return False
        lengths = [v[1]-v[0], v[3]-v[2], v[5]-v[4]]
        if lengths[1] + 1e-10 < min(lengths[0], lengths[2]):
            return False
    return True


def _correction_valid(imp: list[dict], extra: list[dict], direction: int, min_leg: float) -> bool:
    if not 1 <= len(extra) <= 3 or len(imp) != 6:
        return False
    v = [direction * p['value'] for p in imp]
    a = [direction * p['value'] for p in extra]
    # Simple zigzag-type A-B-C only (other corrections are not automatically identified).
    if extra[0]['kind'] != imp[4]['kind'] or v[5]-a[0] < min_leg:
        return False
    if len(a)>=2 and (extra[1]['kind'] != imp[5]['kind'] or not (a[0]<a[1]<v[5])):
        return False
    if len(a)>=3 and (extra[2]['kind'] != imp[4]['kind'] or a[2]>=a[0] or a[1]-a[2]<min_leg):
        return False
    return True


def _invalidation(nodes: list[dict], direction: int, last_label: str) -> dict[str, Any]:
    """One-sided level for the NEXT leg of the *simple* count, not a stop order."""
    lookup={key:node for key,node in zip(['0','1','2','3','4','5','A','B','C'], nodes)}
    conditions={
        '1':('0','2파는 0파의 시작점을 넘어가면 무효'),
        '2':('2','3파가 2파 끝을 반대로 넘어가면 현재 진행 해석 재검토'),
        '3':('1','일반 충격파는 4파가 1파 가격 영역에 진입하면 무효 (대각형 제외)'),
        '4':('4','5파 진행 후보가 4파 끝을 역행하면 재검토'),
        '5':('5','5파 이후 같은 방향의 신저점/신고점은 5파 완료 시점을 재검토'),
        'A':('5','단순 지그재그 B파가 5파 끝을 넘어가면 지그재그 가설 무효 (플랫은 별개)'),
        'B':('B','C파 진행 후보가 B파 끝을 역행하면 현재 단순 지그재그 가설 재검토'),
    }
    if last_label not in conditions:
        return {'price':None,'description':'ABC 완료 이후 새 흐름은 별도 파동으로 분석해야 합니다. 과거 ABC 완료가 미래 추세를 보장하지 않습니다.'}
    key,message=conditions[last_label]
    if key not in lookup:
        return {'price':None,'description':message}
    return {'price':round(float(lookup[key]['value']),4),'description':message}


def _live_invalidation(prices: pd.DataFrame, pivot: dict, seq: list[dict], direction: int) -> dict | None:
    """Validate the *unfinished* leg against subsequent highs/lows.

    Post-C price movement belongs to a NEW sequence: a historical ABC completion
    is not invalidated just because the next trend resumed. This code never
    mislabels a flat or WXY as a verified zigzag.
    """
    tail=prices.iloc[int(pivot['position'])+1:]
    if tail.empty:return None
    labels=['0','1','2','3','4','5','A','B','C']
    last=labels[len(seq)-1]
    if last=='1':basis=seq[0];rule='2파가 0파 시작점 침범';test_direction=-direction
    elif last=='2':basis=seq[2];rule='3파가 2파 끝을 역행';test_direction=-direction
    elif last=='3':basis=seq[1];rule='4파가 1파 영역 침범 (표준 충격파 무효)';test_direction=-direction
    elif last=='4':basis=seq[4];rule='5파가 4파 끝을 역행';test_direction=-direction
    elif last=='5':basis=seq[5];rule='5파 극점 이후 연장: 완료 시점 재선정 필요';test_direction=direction
    elif last=='A':basis=seq[5];rule='B파가 5파 기점을 침범: 단순 지그재그 무효';test_direction=direction
    elif last=='B':basis=seq[7];rule='C파가 B파 끝을 역행: 단순 지그재그 진행 가정 재검토';test_direction=direction
    else:return None  # completed ABC may be historical; staleness is checked elsewhere
    threshold=float(basis['value'])
    offending=(float(tail['High'].max()) if test_direction==1 else float(tail['Low'].min()))
    crossed=(offending>=threshold+1e-8 if test_direction==1 else offending<=threshold-1e-8)
    if not crossed:return None
    return {'rule':rule,'threshold':round(threshold,4),'observed':round(offending,4),
            'last_confirmed':last}


def _candidate_projection(nodes: list[dict], direction: int) -> dict[str,Any]:
    """Mechanical observation zones, not predictions or a calibrated probability."""
    names=[n['파동'] for n in nodes]
    if not names:return {'zone':None,'description':'파동 기준점 없음'}
    last=names[-1]
    values={n['파동']:float(n['가격']) for n in nodes}
    if last=='B':
        # A is opposite to original impulse. Project C = 0.618..1.618 * |A-5| from B.
        leg=values['A']-values['5']
        bounds=sorted((values['B']+leg*.618,values['B']+leg*1.618))
        eps=max(1e-7,abs(values['A'])*1e-5)
        # A simple zigzag C must pass beyond A. Flat/truncated patterns are not inferred.
        if leg>0:bounds[0]=max(bounds[0],values['A']+eps)
        else:bounds[1]=min(bounds[1],values['A']-eps)
        if bounds[0]>=bounds[1]:
            return {'zone':None,'description':'C파가 A파 끝을 넘어가는 유효 참고 구간을 계산할 수 없습니다.'}
        return {'zone':[round(z,4) for z in bounds], 'description':
                'B파 확정 후 A파 길이의 0.618~1.618배를 투영하고 C파가 A파 끝을 넘어야 한다는 지그재그 조건을 적용한 관찰 구간 (내부 5-3-5 미검증, 확률·매수 목표가 아님)'}
    if last=='3':
        size=values['3']-values['2']
        bounds=sorted((values['3']-.5*size,values['3']-.236*size))
        if direction==1:
            bounds=[max(b,values['1']) for b in bounds]
        else:
            bounds=[min(b,values['1']) for b in bounds]
        return {'zone':[round(z,4) for z in sorted(bounds)],'description':
                '3파 길이의 23.6~50% 되돌림 참조 영역(1파 영역 비중첩 제한 적용). 실제 4파 저점/고점 보장 아님'}
    return {'zone':None,'description':'다음 파동의 목표 구간은 아직 산정하지 않습니다.'}


def _result_for_candidate(seq: list[dict], ext: list[dict], direction: int,
                          pivots: list[dict], order: int, invalid_events: list[dict]) -> dict:
    chosen=seq+ext
    labels=['0','1','2','3','4','5','A','B','C'][:len(chosen)]
    nodes=[{'파동':label,'날짜':p['date'],'가격':round(p['value'],4),'종류':'고점' if p['kind']=='H' else '저점',
            'position':p['position'],'value':p['value'],'time':p['time']}
           for label,p in zip(labels,chosen)]
    last=labels[-1]
    if last=='3': current='4파 조정 진행 후보'
    elif last=='4':current='5파 진행 후보'
    elif last=='5':current='A파 조정 진행 후보'
    elif last=='A':current='B파 진행 후보 (지그재그 가정)'
    elif last=='B':current='C파 진행 후보 (지그재그 가정)'
    elif last=='C':current='ABC 완료 가능 · 새 구조 확인 필요'
    else:current='다음 파동 확인 필요'
    phase='상승 충격파' if direction==1 else '하락 충격파'
    inv=_invalidation(chosen,direction,last)
    correction_text=(' / ABC 지그재그 형태 후보 (내부 5-3-5 확인 안 됨)' if ext else '')
    return {'status':'후보','direction':direction,'phase':phase+correction_text,'current':current,
            'nodes':nodes,'anchors':[{'기준':'0파 시작','날짜':nodes[0]['날짜'],'가격':nodes[0]['가격']},
            {'기준':f'마지막 확정 스윙 ({last})','날짜':nodes[-1]['날짜'],'가격':nodes[-1]['가격']}],
            'invalidation':inv,'confirmed_through':nodes[-1]['날짜'], 'order':order,
            'pivot_count':len(pivots), 'invalidated_candidates':len(invalid_events),
            'invalidation_event':None,'correction_status':('ABC 완성 형태' if last=='C' else
                'ABC 진행 후보' if ext else '충격파 이후 조정 또는 진행 후보'),
            'projection':_candidate_projection(nodes,direction),
            'notes':(f'확정 스윙 {len(pivots)}개에서 선택한 **{phase} 구조 후보**입니다. '
                     f'마지막 확인점은 {last}({nodes[-1]["날짜"]})이며 **{current}**로 해석합니다. '
                     'ABC라고 표시해도 A·B·C의 내부 5-3-5 또는 3-3-5 구조를 검증한 것은 아닙니다. '
                     'W-X-Y는 하위 조정 구조가 확인되지 않으면 자동으로 번호를 부여하지 않습니다. '
                     f'스윙은 전후 {order}봉으로 확인하며, 최근 파동 끝은 미확정입니다. '
                     '파동 번호는 예측 확률이나 매매 신호가 아닙니다.')}


def estimate_wave(prices: pd.DataFrame, order: int = 5, lookback: int = 420) -> dict[str, Any]:
    """A single *live* standard impulse or simple ABC candidate and ranked alternatives.

    A,B,C are a STRUCTURAL hypothesis only. Internal 5-3-5 confirmation is not
    available at this resolution, so a zigzag is never described as definitive.
    W-X-Y is deliberately NOT auto-counted without nested corrective subwaves.
    """
    empty={'status':'판정 보류','direction':None,'phase':None,'current':'파동 번호 미확정',
           'nodes':[],'anchors':[],'invalidation':_invalidation([],1,''),
           'notes':'확정된 고점·저점 또는 기본 충격파 규칙을 충족하는 구간이 부족합니다.',
           'confirmed_through':None,'order':order,'pivot_count':0,'invalidated_candidates':0,
           'invalidation_event':None,'alternatives':[],'projection':{'zone':None,'description':'미산정'},
           'correction_status':'미확정','recount_required':False}
    if order not in (3,5,8,12):raise ValueError('민감도는 3, 5, 8, 12 중 선택하세요.')
    if not isinstance(prices,pd.DataFrame) or len(prices)<45 or not {'Low','High','Close'}.issubset(prices.columns):
        return dict(empty)
    candles=prices.dropna(subset=['Low','High','Close'])
    if len(candles)<45:return dict(empty)
    pivots=_alternating_pivots(candles,order,lookback)
    result=dict(empty,pivot_count=len(pivots))
    if len(pivots)<4:
        result['notes']='선택한 민감도에서 확정된 스윙 4개 이상이 필요합니다.'
        return result
    recent=pivots[-24:]
    median_price=float(candles['Close'].tail(40).median())
    threshold=max(.003*median_price,1e-8)
    candidates=[];invalid_events=[]
    for start in range(len(recent)-3):
        for length in (4,5,6):
            if start+length>len(recent):continue
            seq=recent[start:start+length]
            for direction in (1,-1):
                if not _impulse_valid(seq,direction,threshold):continue
                ext=[]
                if length==6:
                    for event in recent[start+length:start+length+3]:
                        trial=ext+[event]
                        if _correction_valid(seq,trial,direction,threshold):ext=trial
                        else:break
                end_index=start+length+len(ext)
                # Never reuse old 0-3 or old ABC with newer confirmed pivots.
                # Their next confirmed swing must be explained by a NEW hypothesis.
                if end_index<len(recent):continue
                end=ext[-1] if ext else seq[-1]
                age=len(candles)-1-int(end['position'])
                if age>max(order*6,30):continue
                if len(ext)==3:
                    if age>max(order*3,9):
                        continue  # completed ABC is history, not current wave
                    # If the entire post-C move already exceeded the old wave 5
                    # in the motive direction, this is a NEW market structure.
                    # The old ABC can remain historical, not an active chart label.
                    after_c=candles.iloc[int(end['position'])+1:]
                    if not after_c.empty:
                        extreme=(float(after_c['High'].max()) if direction==1
                                 else float(after_c['Low'].min()))
                        if ((direction==1 and extreme>float(seq[5]['value'])) or
                            (direction==-1 and extreme<float(seq[5]['value']))):
                            continue
                live_break=_live_invalidation(candles,end,seq+ext,direction)
                if live_break:
                    invalid_events.append(live_break)
                    continue
                score=(int(end['position'])*100+len(seq+ext)*3-int(seq[0]['position'])*.01)
                candidates.append((score,seq,ext,direction))
    if not candidates:
        result['invalidated_candidates']=len(invalid_events)
        result['recount_required']=bool(invalid_events)
        if invalid_events:
            event=invalid_events[-1]
            result['invalidation_event']=event
            result['notes']=(f'기존 파동 후보 {len(invalid_events)}개를 후속 가격이 무효화했습니다. '
                f'최근: {event["rule"]} / 기준 {event["threshold"]:,.2f} / 관측 {event["observed"]:,.2f}. '
                '기존 번호를 지우고 파동 재탐색이 필요합니다.')
        else:
            result['notes']=('현재 확정 스윙에는 활성 상태의 표준 충격파 또는 단순 ABC 후보가 없습니다. '
                '무효·완료된 과거 ABC를 현재 시나리오로 재사용하지 않습니다. '
                '플랫·삼각형·다이아고날·W-X-Y는 확인 없는 자동 확정을 하지 않습니다.')
        return result
    candidates.sort(key=lambda x:x[0],reverse=True)
    chosen=[]
    for score,seq,ext,direction in candidates:
        fingerprint=(direction,seq[0]['position'],seq[-1]['position'],
                     ext[-1]['position'] if ext else None)
        if fingerprint not in [a[0] for a in chosen]:
            chosen.append((fingerprint,seq,ext,direction))
        if len(chosen)>=3:break
    primary=_result_for_candidate(chosen[0][1],chosen[0][2],chosen[0][3],pivots,order,invalid_events)
    primary['alternatives']=[]
    for _,seq,ext,direction in chosen[1:]:
        alt=_result_for_candidate(seq,ext,direction,pivots,order,invalid_events)
        # All share the same data time; these are alternative interpretations,
        # NEVER a probability distribution or independently corroborating votes.
        primary['alternatives'].append(alt)
    primary['recount_required']=False
    return primary


def wave_sensitivity_agreement(prices: pd.DataFrame, orders=(3,5,8)) -> dict[str, Any]:
    """Descriptive robustness check only; agreement is NOT predictive confidence."""
    rows=[]
    for order in orders:
        z=estimate_wave(prices,order=order)
        rows.append({'민감도':order,'상태':z['status'],'구조':z.get('phase') or '미확정',
                     '진행 후보':z['current'],'마지막 기준일':z.get('confirmed_through') or '—'})
    distinct={((z['상태']=='후보'),z['구조'],z['진행 후보'],z['마지막 기준일']) for z in rows}
    return {'rows':rows,'agreement':len(distinct)==1 and all(z['상태']=='후보' for z in rows),
            'notes':'민감도별 파동 해석의 안정성 비교입니다. 일치해도 상승·하락 확률이나 예측 적중률은 아닙니다.'}


def attach_wave_traces(fig: Any, result: dict, cutoff: Any = None, latest: Any = None) -> Any:
    """Plot a SINGLE selected interpretation; distinct colors for motive / ABC.

    Keep incomplete latest movement unnumbered; avoid stacking contradictory
    alternative counts. This only annotates the FIRST chart subplot.
    """
    import plotly.graph_objects as go
    nodes=result.get('nodes') or []
    if not nodes:return fig
    if cutoff is not None:
        nodes=[n for n in nodes if pd.Timestamp(n['time'])>=pd.Timestamp(cutoff)]
    if not nodes:return fig
    for group,color,dash in [({'0','1','2','3','4','5'},'#30bba7','dot'),
                             ({'A','B','C'},'#e7a6ff','solid')]:
        chosen=[n for n in nodes if n['파동'] in group]
        if not chosen:continue
        if 'A' in group and len(chosen):
            prev=next((n for n in reversed(nodes) if n['파동']=='5'),None)
            if prev is not None:chosen=[dict(prev,파동='')]+chosen
        fig.add_trace(go.Scatter(x=[n['time'] for n in chosen],y=[n['가격'] for n in chosen],
            mode='lines+markers+text',text=[n['파동'] for n in chosen],
            textposition=['top center' if n['종류']=='고점' else 'bottom center' for n in chosen],
            textfont={'size':12,'color':'#ffffff'},
            marker={'size':9,'color':color,'line':{'color':'#17212c','width':1.5}},
            line={'color':color,'width':2,'dash':dash},
            name=('충격파 후보 0~5' if '0' in group else 'ABC 조정 형태 후보'),
            showlegend=False, hovertemplate='파동 후보 %{text}<br>%{x}<br>%{y:,.2f}<extra></extra>'),row=1,col=1)
    if latest is not None and nodes:
        t,close=latest
        if pd.Timestamp(t)>pd.Timestamp(nodes[-1]['time']):
            fig.add_trace(go.Scatter(x=[nodes[-1]['time'],t],
                y=[nodes[-1]['가격'],float(close)], mode='lines+markers+text',
                line={'color':'#94a3b8','width':1.3,'dash':'dash'},
                marker={'size':5,'color':'#e2e8f0'},text=['','현재 (미확정)'],
                textposition='top center',textfont={'size':10,'color':'#e2e8f0'},
                name='관측된 최신 가격 (파동 미확정)',showlegend=False,
                hovertemplate='%{x}<br>%{y:,.2f}<extra></extra>'),row=1,col=1)
    return fig
