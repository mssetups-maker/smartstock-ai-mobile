"""Transparent, non-ML chart-formation hypotheses for OHLCV.

Patterns use supplied OHLCV bars and fully confirmed pivots. The most recent supplied candle may be unfinished; callers must flag that possibility.
No chance, win-rate, certainty or investment advice is inferred from a shape.
Pattern price projections are elementary measured moves, NOT model forecasts.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

PATTERN_NAMES = ('상방 돌파', '하방 돌파', '박스권', '이중천장', '이중바닥',
                 '헤드앤숄더', '역헤드앤숄더', '상승 삼각형', '하락 삼각형', '대칭 삼각형')


def _date(t: Any) -> str:
    stamp = pd.Timestamp(t)
    return stamp.strftime('%Y-%m-%d %H:%M') if (stamp.hour or stamp.minute) else stamp.strftime('%Y-%m-%d')


def _prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or not {'High', 'Low', 'Close'}.issubset(prices.columns):
        return pd.DataFrame()
    d = prices.copy().dropna(subset=['High', 'Low', 'Close'])
    for key in ('High','Low','Close'):
        d[key] = pd.to_numeric(d[key], errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['High','Low','Close'])
    d = d[(d['Low'] > 0) & (d['Low'] <= d['High']) & (d['Close'] > 0)]
    return d[~d.index.duplicated(keep='last')].sort_index()


def _atr(d: pd.DataFrame, n: int = 14) -> float:
    c = d['Close'].shift(1)
    true_range = pd.concat([d.High-d.Low, (d.High-c).abs(), (d.Low-c).abs()], axis=1).max(axis=1)
    a = float(true_range.tail(n).median())
    return max(a, float(d.Close.iloc[-1]) * 0.002, 1e-8)


def _pivots(d: pd.DataFrame, span: int = 3, lookback: int = 110) -> list[dict]:
    end=len(d)-span  # exclude non-confirmed last `span` bars
    start=max(span,len(d)-lookback)
    rows=[]
    highs=d['High'].to_numpy(dtype=float)
    lows=d['Low'].to_numpy(dtype=float)
    for k in range(start,end):
        h=highs[k]; l=lows[k]
        ph= h >= highs[k-span:k+span+1].max() and h > highs[k-span:k].max()
        pl= l <= lows[k-span:k+span+1].min() and l < lows[k-span:k].min()
        if ph == pl:  # avoid ambiguous dual pivots and nonpivots
            continue
        rows.append({'index':k,'kind':'H' if ph else 'L',
                     'price':float(h if ph else l),'date':_date(d.index[k]),'time':d.index[k]})
    alternating=[]
    for point in rows:
        if alternating and alternating[-1]['kind']==point['kind']:
            prior=alternating[-1]
            if (point['kind']=='H' and point['price']>prior['price']) or (point['kind']=='L' and point['price']<prior['price']):
                alternating[-1]=point
        else:
            alternating.append(point)
    return alternating


def _record(name, state, direction, d, start_i, level, support, resistance, target, invalid,
            evidence, bar_indices=None, volume_note='', priority=0):
    return {
        'name':str(name), 'state':str(state), 'direction':str(direction),
        'start_date':_date(d.index[start_i]), 'signal_date':_date(d.index[-1]),
        'bars':len(d)-start_i, 'last_close':round(float(d.Close.iloc[-1]),4),
        'level':round(float(level),4), 'support':round(float(support),4),
        'resistance':round(float(resistance),4),
        'target':round(float(target),4) if target is not None and math.isfinite(float(target)) and target>0 else None,
        'invalidation':round(float(invalid),4) if invalid is not None and math.isfinite(float(invalid)) and invalid>0 else None,
        'explanation':str(evidence), 'volume_note':str(volume_note),
        'anchors': [{'date':_date(d.index[i]), 'price':round(float(p),4)} for i,p in (bar_indices or [])],
        '_start_index':int(start_i), '_priority':int(priority),
    }


def _volume_note(d):
    if 'Volume' not in d or len(d)<21: return '거래량 확인 불가'
    v = pd.to_numeric(d.Volume, errors='coerce')
    med = float(v.iloc[-21:-1].median()); last=float(v.iloc[-1])
    if not math.isfinite(last) or med<=0:return '거래량 확인 불가'
    return f'최근 20봉 중앙 거래량의 {last/med:.1f}배'  # zero look-ahead


def _range_patterns(d, atr, recent_bars=3):
    """Independent repeated resistance/support touches are required; never infer proprietary targets."""
    rows=[]
    if len(d)<35:return rows
    last=float(d.Close.iloc[-1]); tol=max(0.65*atr,0.005*last)
    for width in (21,34,55):
        if len(d)<width+5:continue
        formed=d.iloc[-(width+1):-1]
        hi=formed.High.to_numpy(dtype=float)
        lo=formed.Low.to_numpy(dtype=float)
        # 92% quantile suppresses a single extreme wick defining the whole pattern
        r=float(np.quantile(hi,0.92)); s=float(np.quantile(lo,0.08))
        height=r-s
        if height < atr*1.6 or height > last*.32:continue
        hidx=np.where(np.abs(hi-r)<=tol)[0]
        lidx=np.where(np.abs(lo-s)<=tol)[0]
        if len(hidx)<2 or len(lidx)<2:continue
        if int(hidx[-1]-hidx[0]) < max(6,width//5) or int(lidx[-1]-lidx[0]) < max(6,width//5):continue
        close_prev=formed.Close.to_numpy(dtype=float)
        if np.sum(close_prev > r+0.30*atr)>max(1,width//12):continue
        if np.sum(close_prev < s-0.30*atr)>max(1,width//12):continue
        above=last>r+0.25*atr
        below=last<s-0.25*atr
        pending=abs(last-r)<0.8*atr or abs(last-s)<0.8*atr
        if not above and not below and not pending:continue
        name='상방 돌파' if above else '하방 돌파' if below else '박스권'
        state='종가 돌파 확인' if (above or below) else '돌파 대기'
        direction='상승' if above else '하락' if below else '중립'
        level=r if above or (not below and abs(last-r)<abs(last-s)) else s
        target=(r+height) if above else (s-height) if below else None
        invalid=(r-0.6*atr) if above else (s+0.6*atr) if below else None
        evidence=(f'최근 {width}봉에서 저항 {r:,.2f}과 지지 {s:,.2f} 근처에 반복 터치가 나타났습니다. '
                 f'현재 확정 종가 {last:,.2f}은(는) {"저항을 상향" if above else "지지를 하향" if below else "범위 내부에서"} '
                 f'{"돌파했습니다" if (above or below) else "돌파 여부를 기다립니다"}. '
                 '목표가는 박스 높이를 돌파 기준선에 더하거나 뺀 단순 측정치입니다.')
        idx=len(d)-width-1
        points=[(idx+int(hidx[0]),r),(idx+int(lidx[0]),s), (len(d)-1,last)]
        rows.append(_record(name,state,direction,d,idx,level,s,r,target,invalid,evidence,points,
                            _volume_note(d),priority=(120 if (above or below) else 40)+width//5))
    # one best formation per family
    return sorted(rows,key=lambda z:z['_priority'],reverse=True)[:1]


def _double_patterns(d, pts, atr):
    rows=[]
    c=float(d.Close.iloc[-1]); n=len(d); tol=max(1.0*atr,.022*c)
    for a,b,e in zip(pts[:-2],pts[1:-1],pts[2:]):
        if e['kind']!=a['kind'] or b['kind']==a['kind']:continue
        gap=e['index']-a['index']; age=n-1-e['index']
        if gap<8 or gap>65 or age<3 or age>14:continue
        if abs(e['price']-a['price'])>tol:continue
        if a['kind']=='H':
            neck=b['price']; top=(a['price']+e['price'])/2
            if top-neck<1.5*atr:continue
            confirmed=c<neck-.2*atr
            waiting=(c>=neck-.2*atr and c<top)
            if not (confirmed or waiting):continue
            rows.append(_record('이중천장','넥라인 종가 이탈' if confirmed else '넥라인 이탈 대기',
                '하락' if confirmed else '중립',d,a['index'],neck,neck,top,
                neck-(top-neck) if confirmed else None,top+.5*atr if confirmed else None,
                f'유사한 고점 두 개({a["price"]:,.2f}, {e["price"]:,.2f})와 중간 저점(넥라인 {neck:,.2f})이 확인됐습니다. '
                + ('현재 종가가 넥라인 아래입니다.' if confirmed else '넥라인을 종가로 이탈하기 전이라 완성되지 않았습니다.'),
                [(a['index'],a['price']),(b['index'],b['price']),(e['index'],e['price'])],
                _volume_note(d),priority=135 if confirmed else 38))
        else:
            neck=b['price']; bottom=(a['price']+e['price'])/2
            if neck-bottom<1.5*atr:continue
            confirmed=c>neck+.2*atr
            waiting=(c<=neck+.2*atr and c>bottom)
            if not (confirmed or waiting):continue
            rows.append(_record('이중바닥','넥라인 종가 돌파' if confirmed else '넥라인 돌파 대기',
                '상승' if confirmed else '중립',d,a['index'],neck,bottom,neck,
                neck+(neck-bottom) if confirmed else None,bottom-.5*atr if confirmed else None,
                f'유사한 저점 두 개({a["price"]:,.2f}, {e["price"]:,.2f})와 중간 고점(넥라인 {neck:,.2f})이 확인됐습니다. '
                + ('현재 종가가 넥라인 위입니다.' if confirmed else '넥라인을 종가로 돌파하기 전이라 완성되지 않았습니다.'),
                [(a['index'],a['price']),(b['index'],b['price']),(e['index'],e['price'])],
                _volume_note(d),priority=135 if confirmed else 38))
    return sorted(rows,key=lambda z:(z['_priority'],z['_start_index']),reverse=True)[:1]


def _head_shoulders(d, pts, atr):
    rows=[]; n=len(d); close=float(d.Close.iloc[-1])
    for i in range(max(0,len(pts)-15),len(pts)-4):
        seq=pts[i:i+5]; a,b,h,e,r=seq
        if n-1-r['index']>14 or r['index']-a['index']<15:continue
        inv=a['kind']=='L'
        if [p['kind'] for p in seq] != (['L','H','L','H','L'] if inv else ['H','L','H','L','H']):continue
        shoulders=(a['price']+r['price'])/2
        neck=(b['price']+e['price'])/2
        if abs(a['price']-r['price'])>max(1.6*atr,0.035*close):continue
        if abs(b['price']-e['price'])>max(1.8*atr,.035*close):continue
        if inv:
            if shoulders-h['price']<1.0*atr or neck-shoulders<1.0*atr:continue
            yes=close>neck+.2*atr
            possible=h['price']<close<=neck+.2*atr
            target=neck+(neck-h['price']) if yes else None
            invalid=r['price']-.5*atr if yes else None
            name='역헤드앤숄더'
        else:
            if h['price']-shoulders<1.0*atr or shoulders-neck<1.0*atr:continue
            yes=close<neck-.2*atr
            possible=neck-.2*atr<=close<h['price']
            target=neck-(h['price']-neck) if yes else None
            invalid=r['price']+.5*atr if yes else None
            name='헤드앤숄더'
        if not (yes or possible):continue
        rows.append(_record(name,'넥라인 종가 돌파·이탈 확인' if yes else '넥라인 대기',
            '상승' if yes and inv else '하락' if yes else '중립',d,a['index'],neck,
            min(p['price'] for p in seq),max(p['price'] for p in seq),target,invalid,
            f'왼쪽 어깨·머리·오른쪽 어깨와 넥라인 {neck:,.2f}을 식별했습니다. '
            + ('종가 기준 넥라인 확인.' if yes else '아직 종가 기준 넥라인 확인 전입니다.')
            + ' 넥라인은 두 중간 스윙을 단순 평균한 기준입니다.',
            [(p['index'],p['price']) for p in seq],_volume_note(d),priority=150 if yes else 32))
    return sorted(rows,key=lambda z:(z['_priority'],z['_start_index']),reverse=True)[:1]


def _triangles(d, pts, atr):
    rows=[]; n=len(d); close=float(d.Close.iloc[-1])
    highs=[p for p in pts if p['kind']=='H']; lows=[p for p in pts if p['kind']=='L']
    if len(highs)<2 or len(lows)<2:return rows
    for count in (2,):
        h1,h2=highs[-count:]; l1,l2=lows[-count:]
        i=min(h1['index'],l1['index']); j=max(h2['index'],l2['index'])
        if j-i < 12 or j-i>75 or n-1-j>12:continue
        if h2['index']==h1['index'] or l2['index']==l1['index']:continue
        slope_h=(h2['price']-h1['price'])/(h2['index']-h1['index'])
        slope_l=(l2['price']-l1['price'])/(l2['index']-l1['index'])
        h_now=h2['price']+slope_h*(n-1-h2['index'])
        l_now=l2['price']+slope_l*(n-1-l2['index'])
        w0=h1['price']-l1['price']; w1=h_now-l_now
        if w0<=2*atr or w1<=.15*atr or w1>=.88*w0:continue
        flat_h=abs(h2['price']-h1['price'])<=.80*atr
        flat_l=abs(l2['price']-l1['price'])<=.80*atr
        declining=(h2['price'] < h1['price']-.6*atr)
        rising=(l2['price'] > l1['price']+.6*atr)
        if flat_h and rising:name='상승 삼각형'
        elif declining and flat_l:name='하락 삼각형'
        elif declining and rising:name='대칭 삼각형'
        else:continue
        if close > h_now+.22*atr:
            state='저항 추세선 종가 돌파'; direction='상승'; level=h_now
            target=h_now+w0; invalid=l_now
        elif close < l_now-.22*atr:
            state='지지 추세선 종가 이탈'; direction='하락'; level=l_now
            target=l_now-w0;invalid=h_now
        else:
            state='돌파 전 수렴';direction='중립';level=h_now
            target=None;invalid=None
            if close < l_now-.8*atr or close>h_now+.8*atr:continue
        rows.append(_record(name,state,direction,d,i,level,l_now,h_now,target,invalid,
            f'확정 고점 연결선 {h_now:,.2f}, 저점 연결선 {l_now:,.2f}으로 가격 범위가 좁아지는 구간을 식별했습니다. '
            f'패턴 시작 높이 {w0:,.2f}를 투영한 목표값은 참고치이며, 서로 다른 스윙 선택 시 바뀔 수 있습니다.',
            [(p['index'],p['price']) for p in (h1,h2,l1,l2)],
            _volume_note(d),priority=130 if direction!='중립' else 30))
    return rows[:1]


def detect_patterns(prices: pd.DataFrame, pivot_span: int=3, max_patterns: int=3) -> list[dict]:
    """Candidate chart structures based on supplied observed prices, NEVER guaranteed signals.

    Returns up to `max_patterns` dictionaries, newest/strongest first. Price
    formation is a retrospective heuristic, no backtest or calibrated odds.
    """
    if pivot_span not in (3,5,8):raise ValueError('패턴 민감도는 3, 5, 8 중 선택하세요.')
    if max_patterns<1:return []
    d=_prices(prices)
    if len(d)<40:return []
    atr=_atr(d)
    pts=_pivots(d,span=pivot_span,lookback=110)
    rows=[]
    rows.extend(_range_patterns(d,atr))
    rows.extend(_double_patterns(d,pts,atr))
    rows.extend(_head_shoulders(d,pts,atr))
    rows.extend(_triangles(d,pts,atr))
    rows.sort(key=lambda p:(p['_priority'],p['_start_index']),reverse=True)
    # Only report distinct pattern names; do not masquerade overlapping detections as votes.
    seen=set();out=[]
    for p in rows:
        if p['name'] in seen:continue
        seen.add(p['name']);out.append({k:v for k,v in p.items() if not k.startswith('_')})
        if len(out)>=max_patterns:break
    return out


def pattern_description(p: dict) -> str:
    target=f'{p["target"]:,.2f}' if p.get('target') is not None else '미산정(미돌파)'
    invalid=f'{p["invalidation"]:,.2f}' if p.get('invalidation') is not None else '미산정(미돌파)'
    return (f'**{p["name"]} · {p["state"]}** / 방향 후보: {p["direction"]}. '
            f'시작: {p["start_date"]}, 기간 약 {p["bars"]}봉, '
            f'돌파·넥라인 기준 {p["level"]:,.2f}, 측정 목표 {target}, 시나리오 재검토가 {invalid}. '
            f'{p["explanation"]} 거래량: {p["volume_note"]}. '
            '이것은 관측 가격으로부터 계산한 패턴 **후보**이며 정량적 적중 확률이 아닙니다. '
            '예측·자동매수와 독립된 참고 분석입니다.')


def attach_pattern_traces(figure: Any, patterns: list[dict], prices: pd.DataFrame,
                          cutoff: Any = None) -> Any:
    """Annotate first Plotly subplot with pattern reference lines, latest price and label."""
    import plotly.graph_objects as go
    if not patterns:return figure
    d=_prices(prices)
    if d.empty:return figure
    last=d.index[-1]
    for p in patterns[:1]:
        begin=pd.Timestamp(p['start_date'])
        if cutoff is not None and begin < pd.Timestamp(cutoff):
            begin=pd.Timestamp(cutoff)
        if begin>=last:continue
        color='#38d9a9' if p['direction']=='상승' else '#fb7185' if p['direction']=='하락' else '#a5b4fc'
        level=p['level']
        figure.add_trace(go.Scatter(x=[begin,last],y=[level,level],mode='lines',
            name=f'{p["name"]} 기준 {level:,.2f}',line=dict(color=color,width=2,dash='dash'),
            hovertemplate='%{x}<br>패턴 기준: %{y:,.2f}<extra></extra>',showlegend=False),row=1,col=1)
        for label,key in [('참고 목표','target'),('무효화 참고','invalidation')]:
            val=p.get(key)
            if val and math.isfinite(val):
                figure.add_trace(go.Scatter(x=[begin,last],y=[val,val],mode='lines',
                    line=dict(color=color if key=='target' else '#f59e0b',width=1,dash='dot'),
                    name=f'{label} {val:,.2f}',showlegend=False,hovertemplate=f'{label}: %{{y:,.2f}}<extra></extra>'),row=1,col=1)
        # Label uses last bar for horizontal space restraint (mobile).
        figure.add_annotation(x=last,y=level,text=p['name'],showarrow=True,arrowhead=1,
                              ax=-38,ay=-25,font=dict(size=10,color=color),
                              arrowcolor=color,xref='x',yref='y',row=1,col=1)
    return figure
