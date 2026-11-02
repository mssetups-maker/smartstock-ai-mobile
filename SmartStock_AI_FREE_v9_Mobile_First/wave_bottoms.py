"""10.11 research-only Elliott corrective-low *hypotheses*.

No order execution, trained win probabilities, hindsight pivot on unclosed bars,
or assured wave counts. 'Z' is a coarse W-X-Y-X-Z *swing skeleton*;
internal A-B-C subdivisions of W/Y/Z are NOT certified here.
Run on strictly completed OHLC candles only.
"""
from __future__ import annotations

from typing import Any
import math
import pandas as pd
from elliott_waves import _alternating_pivots

STAGES = ('관찰 구간', '초기 반등 관찰', '후행 스윙 저점 확인')


def _data(df: pd.DataFrame) -> pd.DataFrame:
    required = {'Open', 'High', 'Low', 'Close'}
    if not isinstance(df, pd.DataFrame) or not required.issubset(df.columns):
        raise ValueError('Open/High/Low/Close 열이 필요합니다.')
    x = df.sort_index().copy()
    if len(x) < 45 or x.index.has_duplicates:
        raise ValueError('45개 이상의 중복 없는 완성 캔들이 필요합니다.')
    for c in required:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    if x[list(required)].isna().any().any():
        raise ValueError('OHLC 결측 값이 있어 저점 후보를 만들 수 없습니다.')
    if (x['Low'] <= 0).any() or (x['High'] < x['Low']).any() or (x['High'] < x[['Open', 'Close']].max(axis=1)).any() or (x['Low'] > x[['Open', 'Close']].min(axis=1)).any():
        raise ValueError('고가/저가가 잘못된 캔들이 있습니다.')
    return x


def _atr(x: pd.DataFrame) -> float:
    last = x['Close'].shift()
    tr = pd.concat([(x.High-x.Low), (x.High-last).abs(), (x.Low-last).abs()], axis=1).max(axis=1)
    val = float(tr.tail(14).median())
    return max(val, float(x.Close.iloc[-1]) * .003)


def _anchor(p: dict[str, Any], name: str) -> dict[str, Any]:
    return {'label': name, 'date': str(p['date']), 'price': round(float(p['value']), 5),
            'position': int(p['position'])}


def _candidate(x: pd.DataFrame, kind: str, pivots: list[dict], names: list[str],
               zone: tuple[float, float], invalidation: float | None,
               reevaluate: float | None, order: int, details: str,
               min_bottom: float | None = None, max_age: int = 42) -> dict | None:
    latest = len(x) - 1
    last = pivots[-1]
    after = x.iloc[last['position'] + 1:]
    if after.empty or latest - last['position'] > max_age:
        return None
    low = float(after.Low.min())
    low_pos = int(x.index.get_loc(after.Low.idxmin()))
    low_date = str(x.index[low_pos])[:10]
    current = float(x.Close.iloc[-1])
    # Never keep old corrections as current entry candidates.
    if latest - low_pos > max(order*4, 18):
        return None
    if current > float(last["value"]) + _atr(x)*.15:
        return None
    bottom = min(zone); top = max(zone)
    if top <= 0 or top <= bottom:
        return None
    # Stock prices have a strictly positive lower bound. A Fibonacci
    # projection below zero is not a valid quoted buying price.
    bottom = max(bottom, 0.00001)
    if invalidation is not None and low <= invalidation:
        return {'kind':kind, 'status':'무효', 'reason':f'후속 저가 {low:.2f}이(가) 0파 시작 {invalidation:.2f} 이하입니다.',
                'invalidated':True,'anchors':[_anchor(p,n) for p,n in zip(pivots,names)]}
    if reevaluate is not None and low < reevaluate:
        return {'kind':kind, 'status':'재검토', 'reason':f'저가 {low:.2f}이(가) 일반적인 확장 참고선 {reevaluate:.2f} 아래에 있습니다. 연장 조정도 가능하므로 절대 무효화는 아닙니다.',
                'invalidated':False,'anchors':[_anchor(p,n) for p,n in zip(pivots,names)]}
    # A C/Z low must undercut the A or Y low before a tentative COMPLETION label.
    eligible = min_bottom is None or low < min_bottom
    recent = latest - low_pos <= max(order * 3, 9)
    near = bottom <= low <= top or (low > top and (low-top) <= _atr(x)*.7)
    if not near and not (recent and bottom <= low <= top * 1.02):
        return None
    # Earliest turning observation: price left a fresh low and latest close is
    # higher than the preceding close. Never call that a confirmed pivot.
    improving = (recent and eligible and low_pos < latest and
                 current > float(x.Close.iloc[-2]) and current > low + _atr(x)*.12)
    # Strict ex-post confirmation: the low occurred >= order bars ago and
    # is the lowest of a surrounding window, with a fresh local close breakout.
    last_confirmable = low_pos + order <= latest
    left=max(last['position']+1,low_pos-order)
    right=min(latest,low_pos+order)
    local_confirmed = (last_confirmable and low_pos>0 and
                       low <= float(x.Low.iloc[left:right+1].min()) and
                       current > float(x.High.iloc[max(low_pos+1,latest-3):latest].max()))
    confirmed = bool(eligible and recent and local_confirmed)
    stage = STAGES[2] if confirmed else STAGES[1] if improving else STAGES[0]
    # Labels and prices are shown with named provisional/confirmed status,
    # NOT a declared bottom at a date the system could not know then.
    anchors = [_anchor(p,n) for p,n in zip(pivots,names)]
    return {
        'kind':kind,'status':stage,'invalidated':False,'confirmed':confirmed,
        'zone_low':round(bottom,5),'zone_high':round(top,5),
        'observed_low':round(low,5),'observed_low_date':low_date,
        'confirmed_asof':str(x.index[-1])[:10] if confirmed else None,
        'latest_price':round(current,5),
        'invalidation':round(invalidation,5) if invalidation is not None else None,
        'reevaluate':round(reevaluate,5) if reevaluate is not None else None,
        'anchors':anchors,
        'entry_trigger':round(float(x.High.iloc[-2]),5),
        'trigger_description':'다음 완성봉의 종가가 이전 봉 고가를 넘는지 관찰(돌파만으로 매수 승인 아님)',
        'reasons': [details, f'기준 저점 관측 {low:.2f} ({low_date}), 예상 구간 {bottom:.2f}~{top:.2f}',
                    '후행 확정 지점은 좌우 스윙 민감도만큼의 캔들이 지난 후 확인하므로 저점 당일 확정할 수 없습니다.'],
        'research_only':True,
    }


def _sensible_leg(a: dict, b: dict, min_size: float) -> bool:
    return abs(float(a['value'])-float(b['value'])) >= min_size


def detect_bottoms(prices: pd.DataFrame, order: int = 5, lookback: int = 280) -> dict:
    """Return at most one active candidate per 2/C/Z family.

    Always use ONLY price data available in the passed prefix. Confirmation
    is delayed by `order` subsequent bars, never backfilled as an old signal.
    """
    if order not in (3,5,8,12):
        raise ValueError('파동 민감도는 3, 5, 8, 12 중 선택하세요.')
    x=_data(prices)
    pivots=_alternating_pivots(x,order,lookback)
    median=float(x.Close.tail(40).median())
    min_size=max(.012*median, _atr(x)*.8)
    found: dict[str, dict] = {}
    rejected: dict[str, dict] = {}
    for kind, count in (('2파 저점',2),('C파 저점',3),('Z파 저점',5)):
        for end in range(len(pivots),count-1,-1):
            p=pivots[end-count:end]
            if not all(_sensible_leg(a,b,min_size) for a,b in zip(p,p[1:])):
                continue
            if kind=='2파 저점':
                if [i['kind'] for i in p] != ['L','H']: continue
                low0,high1=[float(i['value']) for i in p]
                gain=high1-low0
                if gain<=min_size:continue
                zone=(high1-.786*gain,high1-.5*gain)
                result=_candidate(x,kind,p,['0 (시작)','1 (상승 고점)'],zone,low0,None,order,
                    '상승 1파 뒤 50~78.6% 되돌림 구간. 0파 침범 시 상승 2파 시나리오 무효.')
            elif kind=='C파 저점':
                if [i['kind'] for i in p] != ['H','L','H']: continue
                high0,low_a,high_b=[float(i['value']) for i in p]
                if not low_a < high_b < high0:continue
                drop=high0-low_a
                if drop<=min_size:continue
                zone=(high_b-1.618*drop,high_b-.618*drop)
                result=_candidate(x,kind,p,['A 시작','A 저점','B 고점'],zone,None,
                    high_b-2.0*drop,order,
                    '하락 ABC 지그재그 형태: B 고점에서 A 하락폭의 약 0.618~1.618배 C 관찰 구간. 5-3-5 내부 구조 미확인.',
                    min_bottom=low_a)
            else:
                if [i['kind'] for i in p] != ['H','L','H','L','H']: continue
                h0,lw,hx,ly,hx2=[float(i['value']) for i in p]
                # Strongly downward zigzag-like swing skeleton, not
                # proof of three independent corrective ABC units.
                if not (h0>hx>lw and hx>hx2>ly and ly<lw):continue
                wlen=h0-lw; ylen=hx-ly
                unit=(wlen+ylen)/2
                if unit<=min_size:continue
                zone=(hx2-1.618*unit,hx2-.618*unit)
                result=_candidate(x,kind,p,['W 시작','W 저점','X1 반등','Y 저점','X2 반등'],zone,
                    None,hx2-2.0*unit,order,
                    'W-X-Y-X-Z 형태와 유사한 주요 스윙 골격. W/Y/Z 내부 ABC 확인 없이 진짜 삼중 지그재그로 확정 불가.',
                    min_bottom=ly,max_age=65)
            if result is None:
                continue
            if result.get('status') in ('무효','재검토'):
                if kind not in rejected:rejected[kind]=result
                continue
            found[kind]=result
            break
    return {'status':'ok','asof':str(x.index[-1]), 'order':order,
        'candidates':[found[k] for k in ('2파 저점','C파 저점','Z파 저점') if k in found],
        'rejected':list(rejected.values()), 'pivots_confirmed':len(pivots),
        'notes':('예상 가격대는 피보나치 기반 규칙형 연구 후보입니다. 2파/C파/Z파 번호 및 상승 방향은 확정이 아닙니다. '
                 'W-X-Y-X-Z는 W·Y·Z 각각의 ABC 및 X의 세부 구조를 독립적으로 확인하지 않았으며, 저점 형성 후 신호와 사전 예측을 구분해야 합니다. '
                 '실제 매수·자동 주문 기능은 없습니다.')}


def attach_bottom_traces(fig: Any, report: dict, cutoff=None) -> Any:
    """Overlay actual confirmed anchors + a future-free price band only."""
    import plotly.graph_objects as go
    colors={'2파 저점':'#34d399','C파 저점':'#f59e0b','Z파 저점':'#c084fc'}
    for item in report.get('candidates',[]):
        anchors=item['anchors']
        if cutoff is not None:
            anchors=[a for a in anchors if pd.Timestamp(a['date']) >= pd.Timestamp(cutoff)]
        if anchors:
            fig.add_trace(go.Scatter(x=[a['date'] for a in anchors],
                 y=[a['price'] for a in anchors], mode='lines+markers+text',
                 text=[a['label'] for a in anchors], textposition='top center',
                 line={'color':colors[item['kind']],'dash':'dot','width':1.8},
                 marker={'size':6},name=item['kind']+' 구조 후보',showlegend=False),row=1,col=1)
        fig.add_hrect(y0=item['zone_low'],y1=item['zone_high'],
                      fillcolor=colors[item['kind']],opacity=.09,
                      line_width=0,row=1,col=1,
                      annotation_text=item['kind']+' 관찰 범위',
                      annotation_position='top right')
    return fig
