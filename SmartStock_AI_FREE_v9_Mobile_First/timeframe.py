"""User-selectable charts. Chart timeframe never changes daily-bar AI training."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import clean_prices, indicators, swing_levels

@dataclass(frozen=True)
class Timeframe:
    label: str
    interval: str
    period: str
    description: str

TIMEFRAMES = {
    '1분봉': Timeframe('1분봉','1m','5d','최근 약 5거래일, 공급처 제한'),
    '5분봉': Timeframe('5분봉','5m','1mo','최근 약 1개월'),
    '15분봉': Timeframe('15분봉','15m','1mo','최근 약 1개월'),
    '30분봉': Timeframe('30분봉','30m','1mo','최근 약 1개월'),
    '1시간봉': Timeframe('1시간봉','1h','3mo','최근 약 3개월'),
    '일봉': Timeframe('일봉','1d','10y','최대 10년'),
    '주봉': Timeframe('주봉','1wk','10y','최대 10년'),
    '월봉': Timeframe('월봉','1mo','10y','최대 10년'),
}

def timeframe_config(label: str) -> Timeframe:
    if label not in TIMEFRAMES:
        raise ValueError('지원되지 않는 차트 주기입니다.')
    return TIMEFRAMES[label]

def fetch_timeframe(ticker, label:str) -> pd.DataFrame:
    cfg=timeframe_config(label)
    prices=ticker.history(period=cfg.period, interval=cfg.interval,
                          auto_adjust=True,actions=False,timeout=15)
    data=clean_prices(prices)
    if data.empty or len(data)<10:raise ValueError('해당 주기의 봉 데이터가 충분하지 않습니다.')
    return data

def chart(prices:pd.DataFrame,label:str,ma=True,stochastic=True,trendlines=True,volume=True,
          limit:int=450, ichimoku=False, bollinger=False, daily_trade_levels=None):
    """Draw OHLCV, optional averages, three Slow Stochastics, confirmed pivots.

    Chart uses only bars for the requested timeframe (not daily-level trade targets).
    """
    cfg=timeframe_config(label)
    full=indicators(prices)
    d=full.tail(limit).copy()
    rows=1+int(volume)+3*int(stochastic)
    heights=([0.64] if rows==1 else [0.54])+([0.13] if volume else [])+([0.11]*3 if stochastic else [])
    if rows>1:
        heights=[v/sum(heights) for v in heights]
    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,vertical_spacing=.025,row_heights=heights)
    fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,
      increasing_line_color='#EF6471',decreasing_line_color='#4B91E4',name='캔들'),row=1,col=1)
    colors={5:'#7499D4',10:'#E3B129',20:'#25874B',60:'#93603D',120:'#D23B42',240:'#3268C8'}
    if ma:
        for n,color in colors.items():
            if d[f'MA{n}'].notna().any():
                fig.add_trace(go.Scatter(x=d.index,y=d[f'MA{n}'],mode='lines',name=f'MA{n}',
                                        line=dict(color=color,width=1.6)),row=1,col=1)
    if ichimoku and d['ICH_A'].notna().any() and d['ICH_B'].notna().any():
        fig.add_trace(go.Scatter(x=d.index,y=d.ICH_A,mode='lines',name='일목 선행A (현 위치에 대응)',
                       line=dict(color='#67e8f9',width=1)),row=1,col=1)
        fig.add_trace(go.Scatter(x=d.index,y=d.ICH_B,mode='lines',name='일목 선행B (현 위치에 대응)',
                       line=dict(color='#38bdf8',width=1),fill='tonexty',
                       fillcolor='rgba(56,189,248,.12)'),row=1,col=1)
    if bollinger and d['BB_upper'].notna().any():
        for key in ('BB_upper','BB_lower'):
            fig.add_trace(go.Scatter(x=d.index,y=d[key],mode='lines',name=key,
                            line=dict(color='#a78bfa',width=1,dash='dot')),row=1,col=1)
    if daily_trade_levels is not None and cfg.interval=='1d':
        for key,color in [('entry','#a5b4fc'),('stop','#fb7185'),('tp1','#34d399'),('tp2','#10b981')]:
            price=daily_trade_levels.get(key)
            if price is not None and np.isfinite(price) and price>0:
                fig.add_hline(y=float(price),line_width=1,line_dash='dot',line_color=color,
                              annotation_text=f'{key.upper()} {price:,.2f}',row=1,col=1)
    if trendlines:
        supports,resistances=swing_levels(full)
        for pts,color,name in [(supports, '#33c4a2', '확정 지지 추세선'),
                               (resistances, '#f87171', '확정 저항 추세선')]:
            recent=[p for p in pts if p[0]>=len(full)-limit]
            if len(recent)>=2:
                (i,a),(j,b)=recent[-2:]
                if j>i:
                    projected=b+(b-a)/(j-i)*(len(full)-1-j)
                    fig.add_trace(go.Scatter(x=[full.index[i],full.index[j],full.index[-1]],y=[a,b,projected],
                        name=name,mode='lines',line=dict(color=color,width=1.6,dash='dash')),row=1,col=1)
    row=1
    if volume:
        row+=1
        fig.add_trace(go.Bar(x=d.index,y=d.Volume,name='거래량',marker_color='#64748b',showlegend=False),row=row,col=1)
    if stochastic:
        for n in (5,10,20):
            row+=1
            fig.add_trace(go.Scatter(x=d.index,y=d[f'STO_K{n}'],name=f'{n}K',
                     line=dict(color='#D98B57',width=1.2),showlegend=False),row=row,col=1)
            fig.add_trace(go.Scatter(x=d.index,y=d[f'STO_D{n}'],name=f'{n}D',
                     line=dict(color='#88769B',width=1.2),showlegend=False),row=row,col=1)
            fig.add_hline(y=20,line_width=.6,line_dash='dot',line_color='#888',row=row,col=1)
            fig.add_hline(y=80,line_width=.6,line_dash='dot',line_color='#888',row=row,col=1)
            fig.update_yaxes(range=[0,100],row=row,col=1,title_text=f'{n} Stoch')
    fig.update_layout(template='plotly_dark',height=850 if stochastic else 520,margin=dict(l=10,r=20,t=35,b=10),
                      xaxis_rangeslider_visible=False,legend=dict(orientation='h',y=1.02),hovermode='x unified',
                      title=f'{label} · {cfg.description}')
    if cfg.interval in ('1d','1wk','1mo'):
        fig.update_xaxes(rangebreaks=[dict(bounds=['sat','mon'])])
    return fig
