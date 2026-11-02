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
    # Only provide an unambiguous level for the NEXT hypothetical leg.
    lookup = {k: v for k,v in zip(['0','1','2','3','4','5','A','B','C'],nodes)}
    if last_label in ('1','2'):
        key = '0' if last_label=='1' else '2'
        message = f'{last_label}파 뒤 다음 상승/하락 파동이 {key}번 기점 너머로 움직이면 단순 파동 시나리오 재검토'
    elif last_label=='3':
        key='1'; message='일반 충격파는 4파가 1파 가격 영역에 진입하면 무효(대각형 파동은 별도)'
    elif last_label=='4':
        key='4'; message='5파 후보 진행 중 4파 극점을 반대 방향으로 돌파하면 현재 5파 진행 가정 재검토'
    elif last_label=='A':
        key='5'; message='지그재그 B파가 5파 극점을 넘어가면 해당 단순 ABC 시나리오 재검토'
    else:
        return {'price':None,'description':'단일 가격으로 전체 파동을 무효화할 수 없습니다. 다음 확정 고점·저점을 확인해야 합니다.'}
    return {'price':round(float(lookup[key]['value']),4),'description':message}


def _live_invalidation(prices: pd.DataFrame, pivot: dict, seq: list[dict], direction: int) -> dict | None:
    """Check ALL subsequent wick extremes, not just the closing price.

    Standard impulses only; overlap-allowing diagonals are not modeled.
    No assumption is made about the end of an unfinished corrective wave.
    """
    tail=prices.iloc[int(pivot['position'])+1:]
    if tail.empty:return None
    last=str(len(seq)-1)
    if last=='1':basis=seq[0];rule='2파의 0파 시작점 침범'
    elif last=='2':basis=seq[2];rule='3파 진행 후보의 2파 저점/고점 역행'
    elif last=='3':basis=seq[1];rule='4파가 1파 가격 영역 침범 (일반 충격파 무효)'
    elif last=='4':basis=seq[4];rule='5파 후보의 4파 극점 역행'
    else:return None
    threshold=float(basis['value'])
    offending=tail['Low'].astype(float).min() if direction==1 else tail['High'].astype(float).max()
    crossed=offending<=threshold+1e-9 if direction==1 else offending>=threshold-1e-9
    if not crossed:return None
    return {'rule':rule,'threshold':round(threshold,4),'observed':round(float(offending),4),
            'last_confirmed':last}


def estimate_wave(prices: pd.DataFrame, order: int = 5, lookback: int = 420) -> dict[str, Any]:
    """Return anchored candidate and diagnostic reasons; never force a wave count."""
    empty = {'status':'판정 보류','direction':None,'phase':None,'current':'파동 번호 미확정',
             'nodes':[],'anchors':[],'invalidation':_invalidation([],1,''),
             'notes':'확정된 고점·저점 또는 기본 충격파 규칙을 충족하는 구간이 부족합니다.',
             'confirmed_through':None,'order':order,'pivot_count':0,
             'invalidated_candidates':0,'invalidation_event':None}
    if not isinstance(prices,pd.DataFrame) or len(prices)<45 or not {'Low','High','Close'}.issubset(prices.columns):
        return empty
    if order not in (3,5,8,12):
        raise ValueError('민감도는 3, 5, 8, 12 중 선택하세요.')
    candles=prices.dropna(subset=['Low','High','Close'])
    if len(candles)<45: return empty
    pivots=_alternating_pivots(candles,order,lookback)
    result=dict(empty,pivot_count=len(pivots))
    if len(pivots)<4:
        result['notes']='선택한 민감도에서 확정된 스윙 4개 이상이 필요합니다.'
        return result
    recent=pivots[-24:]
    median_price = float(candles['Close'].tail(40).median())
    move_threshold=max(0.003*median_price, 1e-8)
    candidates=[]
    invalid_events=[]
    for start in range(len(recent)-3):
        for length in (4,5,6):
            if start+length>len(recent):continue
            seq=recent[start:start+length]
            for direction in (1,-1):
                if not _impulse_valid(seq,direction,move_threshold): continue
                ext=[]
                if length==6:
                    for p in recent[start+length:start+length+3]:
                        trial=ext+[p]
                        if _correction_valid(seq,trial,direction,move_threshold):ext=trial
                        else:break
                # Require final hypothesis to reach near current data; don't draw obsolete counts.
                end=ext[-1] if ext else seq[-1]
                bars_since=len(candles)-1-end['position']
                if bars_since>max(order*6,45):continue
                # A confirmed 0-3 sequence is NOT a valid present 4-wave
                # hypothesis if subsequent lows/highs have broken wave 1.
                # Verify the last completed pivot, even if it has not yet
                # produced another confirmed swing.
                live_break=_live_invalidation(candles,end,seq+ext,direction)
                if live_break is not None:
                    invalid_events.append(live_break)
                    continue
                score=(end['position']*10 + (len(seq)+len(ext))*2)
                candidates.append((score,seq,ext,direction))
    if not candidates:
        result['invalidated_candidates']=len(invalid_events)
        if invalid_events:
            event=invalid_events[-1]
            result['invalidation_event']=event
            result['notes']=(f'기존 파동 후보 {len(invalid_events)}개가 후속 캔들에서 무효화됐습니다. '
                 f'최근 사례: {event["rule"]} · 기준 {event["threshold"]:,.2f}, '
                 f'관측 {event["observed"]:,.2f}. 해당 파동 번호는 차트에 표시하지 않습니다.')
            return result
        result['notes']='최근 확정된 스윙에서 2파 되돌림·3파 확장·4파 비중첩 조건을 함께 만족하는 표준 충격파 후보를 찾지 못했습니다. 대각형·복합조정·연장파 등은 자동 판정하지 않습니다.'
        return result
    _, impulse, corr, direction=max(candidates,key=lambda item:item[0])
    chosen=impulse+corr
    labels=['0','1','2','3','4','5','A','B','C'][:len(chosen)]
    nodes=[{'파동':label,'날짜':p['date'],'가격':round(p['value'],4),'종류':'고점' if p['kind']=='H' else '저점',
            'position':p['position'],'value':p['value'],'time':p['time']} for label,p in zip(labels,chosen)]
    last=labels[-1]
    upcoming={'3':'4파 진행 가능','4':'5파 진행 가능','5':'A파 조정 가능',
              'A':'B파 반등 가능','B':'C파 조정 가능','C':'ABC 이후 새 구조 확인 필요'}
    current=upcoming.get(last,'추가 파동 확인 필요')
    # Directional text for 4/5 requires inverse when bearish; keep neutral stage label.
    if last=='3': current='4파 조정 진행 후보'
    if last=='4': current='5파 진행 후보'
    if last=='5': current='A파 조정 진행 후보'
    phase='상승 충격파' if direction==1 else '하락 충격파'
    anchor0=nodes[0]
    inv=_invalidation(chosen,direction,last)
    result.update(status='후보',direction=direction,phase=phase,current=current,
        invalidated_candidates=len(invalid_events),
        nodes=nodes,anchors=[{'기준':'0파 시작','날짜':anchor0['날짜'],'가격':anchor0['가격']},
                             {'기준':'마지막 확정 스윙 ('+last+')','날짜':nodes[-1]['날짜'],'가격':nodes[-1]['가격']}],
        invalidation=inv,confirmed_through=nodes[-1]['날짜'],
        notes=(f'확정 스윙 {len(pivots)}개 중 기본 규칙을 만족하는 **{phase}의 한 가지 후보**입니다. '
               f'마지막 확인 점은 {last}번({nodes[-1]["날짜"]})이며, 이후 **{current}**로 해석할 수 있지만 '
               f'진행 중인 파동의 끝은 아직 확정되지 않았습니다. '
               f'스윙은 좌우 {order}봉을 이용해 확인하므로 최근 {order}봉은 확정 고점·저점으로 처리하지 않습니다. '
               '이 파동 번호는 선택한 시간봉·민감도에 따라 바뀔 수 있고, 정답·예측 확률·매수 신호가 아닙니다.'))
    return result


def attach_wave_traces(fig: Any, result: dict, cutoff: Any = None, latest: Any = None) -> Any:
    """Annotate an already-created Plotly OHLC figure in its FIRST subplot."""
    import plotly.graph_objects as go
    nodes=result.get('nodes') or []
    if not nodes:return fig
    if cutoff is not None:
        nodes=[n for n in nodes if pd.Timestamp(n['time']) >= pd.Timestamp(cutoff)]
        # If only one node within visible chart, no connecting segment.
    if not nodes:return fig
    color='#3ee0ae' if result.get('direction')==1 else '#fba35f'
    fig.add_trace(go.Scatter(x=[n['time'] for n in nodes],y=[n['가격'] for n in nodes],
          mode='lines+markers+text',text=[n['파동'] for n in nodes],
          textposition=['top center' if n['종류']=='고점' else 'bottom center' for n in nodes],
          textfont={'size':12,'color':'#fff'},
          marker={'size':9,'color':color,'line':{'color':'#1e293b','width':2}},
          line={'color':color,'width':2.7,'dash':'dot'},
          name='엘리엇 파동 후보 (확정 스윙)',showlegend=True,
          hovertemplate='파동 후보 %{text}<br>%{x}<br>%{y:,.2f}<extra></extra>'),row=1,col=1)
    if latest is not None and len(nodes):
        current_time, current_price = latest
        # Use observed price, never a guessed wave endpoint.
        if pd.Timestamp(current_time) > pd.Timestamp(nodes[-1]['time']):
            fig.add_trace(go.Scatter(x=[nodes[-1]['time'],current_time],
                y=[nodes[-1]['가격'],float(current_price)],mode='lines+markers+text',
                line={'color':color,'width':1.5,'dash':'dash'},
                marker={'size':5,'color':'#e2e8f0'},
                text=['','현재 (미확정)'],textposition='top center',
                textfont={'size':10,'color':'#e2e8f0'},
                name='최근 실제 가격 구간 (파동 미확정)',showlegend=False,
                hovertemplate='%{x}<br>%{y:,.2f}<extra></extra>'),row=1,col=1)
    return fig
