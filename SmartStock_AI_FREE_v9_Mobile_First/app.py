"""SmartStock AI Dashboard, research-only Streamlit app."""
from __future__ import annotations

import datetime as dt
import math
import re
import sqlite3
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from engine import HORIZON, FEATURE_LABELS, analyze, clean_prices, indicators, rule_summary, swing_levels, validate_horizon
from options_engine import analyze_options, pick_expiry
from intelligence import fundamentals_snapshot, news_headline_score, build_llm_context
from research import adaptive_signals, wave_candidates, free_explanation
from local_ai import explain_with_ollama
from journal import update_journal
from strict_validation import audit_history
from risk_engine import data_audit, risk_report, hypothetical_size, historical_stress, compare_market
from point_in_time import read_snapshots,join_observations
from context_model import assess_context
from sec_history import get_companyfacts,extract_filing_series,financial_features_asof
from sec_backfill import save_sec_filings,load_sec_filings
from historical_backfill import backfill_news
from timeframe import TIMEFRAMES, fetch_timeframe, chart as multi_chart
from mobile_ui import resolve_known_ticker, mobile_page
from v8_features import (reliability_calibration, classify_market, model_cost_stress,
    option_snapshot_changes, earnings_event_info, indicator_candidate_lab, watchlist_relative,
    interval_coverage_audit, forecast_error_patterns)
import os

st.set_page_config(page_title='SmartStock AI 9.0 | 모바일', page_icon='📈', layout='centered', initial_sidebar_state='collapsed')
st.markdown("""<style>
/* 모바일 우선: 크고 명확한 터치 대상, 불필요한 좌우 공간 제거 */
.block-container{padding-top:.55rem;padding-bottom:5rem;max-width:780px}
h1{font-size:1.48rem!important;line-height:1.25!important;margin-bottom:.15rem!important}
h2{font-size:1.23rem!important}
h3{font-size:1.08rem!important}
[data-testid="stMetricLabel"]{font-size:.83rem}
[data-testid="stMetricValue"]{font-size:1.35rem;overflow-wrap:anywhere}
.stButton button,.stFormSubmitButton button,.stDownloadButton button{min-height:46px;border-radius:13px;font-weight:650}
div[data-baseweb="select"]>div{min-height:46px;border-radius:12px}
input[type="text"]{min-height:42px;font-size:16px!important}
[data-testid="stVerticalBlock"]{gap:.52rem}
[data-testid="stExpander"]{border-radius:12px}
[data-testid="stPlotlyChart"]{max-width:100%;overflow:hidden}
[data-testid="stDataFrame"]{max-width:100%;overflow:auto}
@media(max-width:640px){
  .block-container{padding-left:.78rem;padding-right:.78rem;padding-top:.42rem}
  [data-testid="stMetricValue"]{font-size:1.21rem}
  [data-testid="stHorizontalBlock"]{gap:.43rem}
  [data-testid="stMarkdownContainer"] p{line-height:1.45}
  [data-testid="stAlert"]{padding:.6rem .7rem}
  [data-testid="stCaptionContainer"]{font-size:.79rem}
}
</style>""", unsafe_allow_html=True)

POPULAR = {
    'NVDA':'엔비디아', 'MSFT':'마이크로소프트', 'AAPL':'애플',
    'GOOGL':'알파벳', 'AMZN':'아마존', 'META':'메타',
    'TSLA':'테슬라', 'AMD':'AMD', 'PLTR':'팔란티어', 'CRWV':'코어위브',
    'AVGO':'브로드컴', 'SOFI':'소파이', 'COIN':'코인베이스',
    'QQQ':'나스닥100 ETF', 'SPY':'S&P500 ETF',
    '005930.KS':'삼성전자', '000660.KS':'SK하이닉스'
}
MA_COLORS = {5:'#7499D4',10:'#E3B129',20:'#25874B',60:'#93603D',120:'#D23B42',240:'#3268C8'}
DB = Path(__file__).with_name('prediction_log.sqlite3')
JOURNAL = Path(__file__).with_name('data')/'predictions.csv'
CONTEXT_STORE = Path(__file__).with_name('data')/'context_snapshots.jsonl'

@st.cache_data(ttl=3600,show_spinner=False)
def do_context_assessment(prices, archive, horizon, sec_features=None):
    return assess_context(prices,archive,horizon,sec_features=sec_features)

@st.cache_data(ttl=86400,show_spinner=False)
def load_sec_data(symbol):
    return extract_filing_series(get_companyfacts(symbol))

@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(symbol):
    return fundamentals_snapshot(yf.Ticker(symbol).info)

@st.cache_data(ttl=1800, show_spinner=False)
def load_options_chain(symbol):
    t=yf.Ticker(symbol)
    exp=pick_expiry(list(t.options))
    if exp is None:raise ValueError('선택한 종목에 조회 가능한 미래 만기 옵션이 없습니다.')
    chain=t.option_chain(exp)
    return exp,chain.calls,chain.puts

@st.cache_data(ttl=3600, show_spinner=False)
def load_macro():
    out={}
    for sym,name in [('SPY','S&P500 ETF'),('QQQ','나스닥100 ETF'),('^VIX','VIX 변동성지수'),('^TNX','미국 10년물 수익률 지수')]:
        try:
            frame=yf.Ticker(sym).history(period='2mo',interval='1d',timeout=10)
            series=frame['Close'].dropna()
            if len(series)>=6:
                out[name]={'last':round(float(series.iloc[-1]),2),
                           'five_day_pct':round(float((series.iloc[-1]/series.iloc[-6]-1)*100),2),
                           'asof':series.index[-1].strftime('%Y-%m-%d')}
        except Exception:
            continue
    return out


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbol:str):
    hist = yf.Ticker(symbol).history(period='10y', interval='1d', auto_adjust=True, actions=False, timeout=15)
    return clean_prices(hist)

@st.cache_data(ttl=3600, show_spinner=False)
def do_analyze(raw, horizon_days):
    df=indicators(raw)
    res=analyze(df, horizon_days=horizon_days)
    return df,res

@st.cache_data(ttl=3600, show_spinner=False)
def do_strict_audit(raw, horizon_days, latest_score):
    return audit_history(raw, horizon_days, latest_score)

@st.cache_data(ttl=900, show_spinner=False)
def load_intraday(symbol):
    t=yf.Ticker(symbol)
    return clean_prices(t.history(period='5d',interval='15m',auto_adjust=True,actions=False,timeout=15))

@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark(symbol):
    return clean_prices(yf.Ticker(symbol).history(period='10y',interval='1d',auto_adjust=True,actions=False,timeout=15))

@st.cache_data(ttl=150, show_spinner=False)
def load_selected_timeframe(symbol, label):
    return fetch_timeframe(yf.Ticker(symbol), label)

@st.cache_data(ttl=21600,show_spinner=False)
def load_earnings_events(symbol):
    return earnings_event_info(yf.Ticker(symbol).calendar)

@st.cache_data(ttl=900,show_spinner=False)
def load_screener_prices(symbol):
    return clean_prices(yf.Ticker(symbol).history(period='1y',interval='1d',auto_adjust=True,
                                                  actions=False,timeout=12))

@st.cache_data(ttl=900, show_spinner=False)
def lookup(search):
    s=yf.Search(search, max_results=10, news_count=0, raise_errors=False)
    results=[]
    for x in s.quotes or []:
        symbol=x.get('symbol')
        if symbol and x.get('quoteType','').lower() in ('equity','etf','index',''):
            results.append((symbol, x.get('shortname') or x.get('longname') or symbol))
    return results

@st.cache_data(ttl=900, show_spinner=False)
def latest_news(symbol):
    try:
        news=(yf.Search(symbol, news_count=7, max_results=0, raise_errors=False).news or [])[:7]
        if news:return news
    except Exception:pass
    # Open RSS headline fallback, only headlines and public links (no article scraping).
    try:
        q=urllib.parse.quote(f'{symbol} stock')
        url=f'https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en'
        req=urllib.request.Request(url,headers={'User-Agent':'SmartStockResearch/5.0'})
        with urllib.request.urlopen(req,timeout=8) as response:root=ET.fromstring(response.read(250000))
        news=[]
        for item in root.findall('.//channel/item')[:7]:
            news.append({'title':item.findtext('title') or '',
                         'link':item.findtext('link') or '',
                         'publisher':item.findtext('source') or 'Google News RSS'})
        return news
    except Exception:return []


@st.cache_data(ttl=3600, show_spinner=False)
def options_term_structure(symbol,spot):
    t=yf.Ticker(symbol)
    expiries=list(t.options)
    today=dt.date.today()
    selected=[]
    for n in (7,30,60):
        valid=[(abs((dt.date.fromisoformat(e)-today).days-n),e) for e in expiries if dt.date.fromisoformat(e)>today]
        if valid:
            exp=min(valid)[1]
            if exp not in selected:selected.append(exp)
    result=[]
    for exp in selected:
        try:
            pair=t.option_chain(exp)
            rep=analyze_options(spot,pair.calls,pair.puts,exp)
            result.append({'만기':exp,'잔존일':rep.dte,'ATM_IV':rep.implied_vol,
                           '예상변동폭':rep.implied_move_pct,'OI_PCR':rep.oi_pcr})
        except Exception:continue
    return pd.DataFrame(result)


def readable_news(news):
    if not isinstance(news,dict):return None
    item=news.get('content') or news
    if not isinstance(item,dict):return None
    title=item.get('title') or news.get('title')
    source=(item.get('provider') or {}).get('displayName','') if isinstance(item.get('provider'),dict) else item.get('publisher','')
    uri=item.get('canonicalUrl') or item.get('clickThroughUrl') or item.get('link') or news.get('link')
    if isinstance(uri,dict):uri=uri.get('url')
    return title,uri,source


def price_str(p,symbol):
    if p is None or not np.isfinite(p):return '—'
    if symbol.endswith(('.KS','.KQ')):return f'₩{p:,.0f}'
    if symbol.endswith('.T'):return f'¥{p:,.0f}'
    return f'${p:,.2f}'


def persist_prediction(symbol, raw, p, horizon_days):
    """Track settled outcomes per (symbol, date, horizon), avoiding label mixing."""
    try:
        h=validate_horizon(horizon_days)
        with sqlite3.connect(DB, timeout=3) as con:
            con.execute("CREATE TABLE IF NOT EXISTS predictions_periods ("
                "symbol TEXT, price_date TEXT, horizon_days INTEGER, price REAL, "
                "p_up REAL, outcome REAL, correct INTEGER, "
                "PRIMARY KEY(symbol,price_date,horizon_days))")
            rows=con.execute('SELECT price_date,price,p_up FROM predictions_periods '
                             'WHERE symbol=? AND horizon_days=? AND outcome IS NULL',(symbol,h)).fetchall()
            for day,ref,score in rows:
                at=raw.index.searchsorted(pd.Timestamp(day))
                if at<len(raw) and raw.index[at].strftime('%Y-%m-%d')==day and at+h<len(raw):
                    actual=float(raw.Close.iloc[at+h]/ref-1)
                    correct=int((score>=.5)==(actual>0))
                    con.execute('UPDATE predictions_periods SET outcome=?,correct=? '
                                'WHERE symbol=? AND price_date=? AND horizon_days=?',
                                (actual,correct,symbol,day,h))
            day=raw.index[-1].strftime('%Y-%m-%d')
            con.execute('INSERT OR IGNORE INTO predictions_periods '
                        '(symbol,price_date,horizon_days,price,p_up) VALUES(?,?,?,?,?)',
                        (symbol,day,h,float(raw.Close.iloc[-1]),float(p)))
            total,hits=con.execute('SELECT COUNT(*),AVG(correct) FROM predictions_periods '
                                   'WHERE symbol=? AND horizon_days=? AND correct IS NOT NULL',(symbol,h)).fetchone()
            return int(total),hits
    except (sqlite3.Error,PermissionError,OSError):
        return 0,None


def trend_line(fig,df,points,positive,color):
    if len(points)<2:return
    p1,p2=points[-2],points[-1]
    if p1[0]==p2[0]:return
    slope=(p2[1]-p1[1])/(p2[0]-p1[0])
    xs=[df.index[p1[0]],df.index[p2[0]],df.index[-1]]
    ys=[p1[1],p2[1],p1[1]+slope*(len(df)-1-p1[0])]
    # Trendline labels are based on observed swing direction, not presumed direction.
    label = ('상승 지지선' if slope >= 0 else '하락 지지선') if positive else ('상승 저항선' if slope >= 0 else '하락 저항선')
    fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',name=label,
          line=dict(color=color,dash='dash',width=2)),row=1,col=1)


def main_chart(df,levels,symbol,days):
    d=df.tail(days).copy()
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.025,
                      row_heights=[.57,.143,.143,.144])
    fig.add_trace(go.Candlestick(x=d.index,open=d['Open'],high=d['High'],low=d['Low'],close=d['Close'],
                        increasing_line_color='#EF6471',decreasing_line_color='#4B91E4',name='캔들'),row=1,col=1)
    for n,color in MA_COLORS.items():
        if d[f'MA{n}'].notna().any():
            fig.add_trace(go.Scatter(x=d.index,y=d[f'MA{n}'],mode='lines',name=f'MA {n}',
                                     line=dict(color=color,width=2 if n>=20 else 1)),row=1,col=1)
    # Confirmed swing pivots: only points whose right-side confirmation bars exist.
    full_sup,full_res=swing_levels(df)
    for pivots,up,color in [(full_sup,True,'#33c4a2'),(full_res,False,'#ed7272')]:
        if len(pivots)>=2:
            recent=[p for p in pivots if p[0]>=len(df)-days]
            if len(recent)>=2: trend_line(fig,df,recent,up,color)
    for n,color,dash in [('entry','#a5b4fc','dot'),('stop','#fb7185','dash'),('tp1','#34d399','dash'),('tp2','#10b981','dot')]:
        v=levels[n]
        fig.add_hline(y=v,line_color=color,line_dash=dash,line_width=1,
                      annotation_text=f'{n.upper()} {v:,.2f}',annotation_position='right',row=1,col=1)
    for idx,n in enumerate([5,10,20],start=2):
        k,dname=f'STO_K{n}',f'STO_D{n}'
        fig.add_trace(go.Scatter(x=d.index,y=d[k],line=dict(color='#D98B57',width=1.5),name=f'{n} %K',showlegend=False),row=idx,col=1)
        fig.add_trace(go.Scatter(x=d.index,y=d[dname],line=dict(color='#88769B',width=1.5),name=f'{n} %D',showlegend=False),row=idx,col=1)
        for v in [20,80]: fig.add_hline(y=v,line_color='#888',line_dash='dot',line_width=.5,row=idx,col=1)
        fig.update_yaxes(range=[-3,103],row=idx,col=1,title_text={2:'단기',3:'중기',4:'장기'}[idx])
    fig.update_layout(template='plotly_dark',height=920,margin=dict(l=10,r=35,t=20,b=8),
         legend=dict(orientation='h',y=1.035,x=0),xaxis_rangeslider_visible=False,
         hovermode='x unified',font=dict(size=11))
    fig.update_xaxes(rangebreaks=[dict(bounds=['sat','mon'])])
    fig.update_yaxes(tickformat=',.2f',row=1,col=1)
    return fig


def forecast_chart(paths,spot,horizon_days):
    x=list(range(horizon_days+1))
    fig=go.Figure()
    lo=spot*(1+paths['low']); med=spot*(1+paths['median']); hi=spot*(1+paths['high'])
    fig.add_trace(go.Scatter(x=x,y=hi,mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x,y=lo,mode='lines',line=dict(width=0),fill='tonexty',
                              fillcolor='rgba(99,102,241,0.22)',name='과거 유사구간 10~90% 범위'))
    fig.add_trace(go.Scatter(x=x,y=med,mode='lines',line=dict(color='#a3e635',width=3),name='중앙 시나리오'))
    fig.add_hline(y=spot,line_color='#aaa',line_dash='dash')
    fig.update_layout(template='plotly_dark',height=330,margin=dict(l=5,r=5,t=22,b=18),
          xaxis_title=f'앞으로의 거래일 수 (선택 {horizon_days}거래일)',yaxis_title='예상 가격(원래 통화)',
          legend=dict(orientation='h',y=1.2),hovermode='x unified')
    return fig

# ---- Mobile-first UI: no sidebar dependency ----
st.markdown('## 📈 SmartStock AI <span style="color:#59a7fa">9.0</span>', unsafe_allow_html=True)
st.caption('휴대폰용 분석 화면 · 매수/매도 자동 실행 없음')
if 'symbol' not in st.session_state:
    st.session_state.symbol='NVDA'

# One visible search box, preset tickers and a simple two-step company search.
with st.form('mobile_stock_search', clear_on_submit=False):
    typed=st.text_input('🔎 종목 검색', placeholder='예: 코어위브, 엔비디아, CRWV, 005930.KS',
                        label_visibility='visible', key='mobile_search_text')
    search_pressed=st.form_submit_button('🔎 종목 찾기 / 열기', use_container_width=True)
if search_pressed and typed.strip():
    term=typed.strip()
    direct=resolve_known_ticker(term,POPULAR)
    if direct:
        st.session_state.symbol=direct
        st.session_state.pop('mobile_search_results',None)
        st.rerun()
    else:
        try:
            matches=lookup(term)
            if matches:
                st.session_state.mobile_search_results=matches
            else:
                st.session_state.mobile_search_results=[]
                st.warning('검색 결과가 없습니다. 아래 직접 티커 입력으로 시도할 수 있습니다.')
        except Exception:
            st.session_state.mobile_search_results=[]
            st.warning('종목 검색 연결이 불안정합니다. 정확한 종목 티커를 직접 입력해 주세요.')
if st.session_state.get('mobile_search_results'):
    candidates=st.session_state.mobile_search_results
    chosen=st.selectbox('검색 결과 중 선택', candidates,
        format_func=lambda a:f'{a[1]} ({a[0]})',key='mobile_search_pick')
    if st.button('✅ 선택한 주식 차트 열기',use_container_width=True):
        st.session_state.symbol=chosen[0]
        st.session_state.pop('mobile_search_results',None)
        st.rerun()

quick=st.columns(4,gap='small')
for column,tk in zip(quick,['CRWV','NVDA','TSLA','AAPL']):
    with column:
        if st.button(tk,key=f'mobile_quick_{tk}',use_container_width=True):
            st.session_state.symbol=tk
            st.session_state.pop('mobile_search_results',None)
            st.rerun()

choices={'1일':1,'1주':5,'2주':10,'1개월':21,'2개월':42,'3개월':63,
         '6개월':126,'1년':252,'직접 입력':None}
form_left,form_right=st.columns(2,gap='small')
with form_left:
    sel=st.selectbox('🔮 AI 예측 기간',list(choices),index=3,key='mobile_horizon_choice')
with form_right:
    chart_interval=st.selectbox('🕯️ 차트 주기',list(TIMEFRAMES),index=5,key='mobile_chart_interval')
if choices[sel] is None:
    horizon_days=int(st.number_input('직접 입력: 앞으로 몇 거래일?',min_value=1,max_value=252,value=30,step=1))
else:
    horizon_days=choices[sel]
horizon_days=validate_horizon(horizon_days)

with st.expander('⚙️ 더 많은 설정 · 데이터 갱신 · CSV · 오프라인 시연',expanded=False):
    st.caption('한국주식: 삼성전자 005930.KS, SK하이닉스 000660.KS')
    manual_ticker=st.text_input('검색이 안 될 때 직접 티커 입력',value='',placeholder='예: AMD, 005930.KS',key='mobile_manual_ticker')
    if st.button('직접 입력한 티커 열기',use_container_width=True):
        ticker=manual_ticker.upper().strip()
        if re.fullmatch(r'[A-Z0-9.\-\^=]{1,22}',ticker):
            st.session_state.symbol=ticker
            st.rerun()
        else:st.error('영문 티커 또는 005930.KS 형식으로 입력하세요.')
    chart_days=st.slider('차트 표시 봉 개수',60,500,120,20, key='mobile_chart_bars')
    do_refresh=st.button('🔄 최신 데이터 다시 조회',use_container_width=True)
    uploaded=st.file_uploader('내 가격 CSV 분석 (선택)',type='csv')
    use_demo=st.checkbox('인터넷 없이 가상 데이터로 화면 시험해 보기',value=False)
    st.caption('무료 분봉은 지연되거나 과거 조회가 제한될 수 있습니다. AI 예측 모델은 일봉 기준입니다.')

if do_refresh:
    load_prices.clear()
    load_benchmark.clear()
    do_analyze.clear()
    do_strict_audit.clear()
    latest_news.clear()
    load_fundamentals.clear()
    load_options_chain.clear()
    load_macro.clear()
    load_intraday.clear()
    load_selected_timeframe.clear()
    load_screener_prices.clear()
    options_term_structure.clear()
    do_context_assessment.clear()
    st.toast('가격 캐시를 지우고 새로 분석합니다.')

symbol=st.session_state.symbol
is_demo=False
try:
    if uploaded is not None:
        raw_df=pd.read_csv(uploaded)
        date_col=next((c for c in raw_df.columns if c.lower() in ('date','datetime','날짜')),None)
        if date_col is None: raise ValueError('CSV에 Date 열이 필요합니다.')
        raw_df.index=pd.to_datetime(raw_df.pop(date_col),errors='coerce')
        raw=clean_prices(raw_df)
        source='업로드한 CSV'
    elif use_demo:
        raw_df=pd.read_csv(Path(__file__).with_name('demo_synthetic.csv'),parse_dates=['Date'],index_col='Date')
        raw=clean_prices(raw_df)
        source='가상의 시뮬레이션 데이터'
        is_demo=True
    else:
        with st.spinner(f'{symbol} 과거 시세 불러오는 중...'):
            raw=load_prices(symbol)
        source='Yahoo Finance 일봉 (지연 가능)'
    if len(raw)<90:raise ValueError('분석 가능한 일봉 데이터가 90개 미만입니다. 티커 또는 CSV를 확인하세요.')
except Exception as ex:
    st.error(f'가격 데이터를 가져오지 못했습니다: {ex}')
    st.info('미국주식은 NVDA, CRWV / 한국주식은 005930.KS 형태로 입력해 보세요. 무료 데이터 공급 오류 시 CSV 업로드나 가상 데이터 시연을 이용할 수 있습니다.')
    st.stop()

with st.spinner('지표 계산·모델 학습·백테스트 진행 중...'):
    df,res=do_analyze(raw,horizon_days)
    strict=do_strict_audit(raw,horizon_days,res.probability)
    res.recommendation=strict.recommendation
    res.explanation=strict.explanation
    adapted=adaptive_signals(df,horizon_days=horizon_days)
    waves=wave_candidates(df)
    health=data_audit(raw)
    risk=risk_report(raw,res.levels)
    if health.status=='차단' or is_demo or uploaded is not None:
        strict.recommendation='판단 보류'
        strict.explanation='가상/업로드 데이터이거나 최신 가격 검증을 통과하지 못해 현재 매수 판단을 보류합니다.'
    res.recommendation=strict.recommendation
    res.explanation=strict.explanation

spot=float(raw['Close'].iloc[-1]); date=raw.index[-1].strftime('%Y-%m-%d')
price_delta=float(raw['Close'].iloc[-1]/raw['Close'].iloc[-2]-1)*100
st.markdown(f'### {"🧪 시연용 가상 종목" if is_demo else POPULAR.get(symbol,symbol)} · {symbol}')
st.caption(f'{date} 기준 확정 일봉 · 출처: {source} · 예측 {horizon_days}거래일')
if is_demo:
    st.warning('시연용 가상 가격입니다. 실제 종목 분석으로 보지 마세요.')
if health.status=='차단' and not is_demo:
    st.error('데이터 품질 확인 필요 — 매수 판단 자동 보류')
elif health.status=='주의' and not is_demo:
    st.warning('데이터 날짜 또는 품질에 주의가 필요합니다.')

summary_left,summary_right=st.columns(2,gap='small')
summary_left.metric('현재 확인된 종가',price_str(spot,symbol),f'{price_delta:+.2f}% 전일 대비')
summary_right.metric('AI 상승 방향 점수',
    f'{res.probability*100:.1f}점' if res.probability is not None else '학습 부족')

# No investment recommendations from unverified scores.
if strict.recommendation=='판단 보류':
    st.info(f'**AI 판단: 판단 보류** · {horizon_days}거래일 예측')
else:
    st.success(f'**검증 기반 참고 신호: {strict.recommendation}** · {horizon_days}거래일')
st.caption('상승점수는 실제 수익 확률이 아닙니다. 예측의 정확성 및 수익 보장 없음.')
regime8=classify_market(raw)

# Mobile primary navigation: just four large finger-friendly choices.
section_top=st.segmented_control('👇 분석 화면 선택',
    ['📈 차트','🔮 예측','📌 옵션','☰ 더보기'],
    default='📈 차트',key='mobile_section_top',selection_mode='single') or '📈 차트'
section_detail=None
if section_top=='☰ 더보기':
    section_detail=st.selectbox('보고 싶은 분석',
        ['🛡 위험관리','📰 뉴스','🏢 재무·시장','🧩 AI 종합 해설','🧠 학습·검증','🔎 관심종목 비교'],
        key='mobile_section_detail')
active_section=mobile_page(section_top,section_detail)

if active_section=='chart':
    st.caption('차트를 손가락으로 밀거나 확대할 수 있습니다. 봉 주기는 화면 위쪽에서 바꿉니다.')
    st.markdown('#### 자동 목표가 · 손절 참고가격')
    topa,topb=st.columns(2,gap='small')
    topa.metric('1차 목표 (TP1)',price_str(res.levels['tp1'],symbol))
    topb.metric('손절 참고가 (SL)',price_str(res.levels['stop'],symbol))
    st.caption(f'2차 목표가 {price_str(res.levels["tp2"],symbol)} · 롱 포지션 가정, 자동 주문 아님')

if active_section in ('forecast','model'):
    with st.expander('📋 AI 과거 검증 결과 확인',expanded=False):
        a,b=st.columns(2)
        a.metric('시간순 과거 검증',f'{strict.evaluations}건')
        b.metric('매수 신호 발생',f'{strict.evaluated_long_signals}건')
        a,b=st.columns(2)
        a.metric('과거 순방향 적중률',f'{strict.long_hit_rate:.1%}' if strict.long_hit_rate is not None else '표본 부족')
        b.metric('95% 통계적 하한',f'{strict.long_wilson_lower:.1%}' if strict.long_wilson_lower is not None else '표본 부족')
        st.caption(res.explanation)
        if strict.diagnostics:st.caption('판단 보류 사유: '+' / '.join(strict.diagnostics[:4]))

if active_section=='chart':
    with st.expander('⚙️ 차트 보조지표 표시 설정',expanded=False):
        aa,bb=st.columns(2)
        show_ma=aa.checkbox('이동평균선',value=True,key='chart_ma')
        show_stoch=bb.checkbox('스토캐스틱 3개',value=True,key='chart_stoch')
        show_trend=aa.checkbox('자동 추세선',value=True,key='chart_trends')
        show_volume=bb.checkbox('거래량',value=True,key='chart_vol')
        show_cloud=aa.checkbox('일목 구름',value=True,key='chart_cloud')
        show_bb=bb.checkbox('볼린저',value=False,key='chart_bollinger')
    display_interval=chart_interval
    if (is_demo or uploaded is not None) and display_interval!='일봉':
        st.warning('가상/업로드 CSV는 일봉만 포함합니다. 다른 주기의 가격을 만들어내지 않고 일봉을 표시합니다.')
        display_interval='일봉'
    try:
        if display_interval=='일봉' or is_demo or uploaded is not None:
            shown=raw
        else:
            shown=load_selected_timeframe(symbol,display_interval)
        figure=multi_chart(shown,display_interval,ma=show_ma,stochastic=show_stoch,
                      trendlines=show_trend,volume=show_volume,limit=chart_days,
                      ichimoku=show_cloud,bollinger=show_bb,
                      daily_trade_levels=res.levels if display_interval=='일봉' else None)
        figure.update_layout(height=700 if show_stoch else 470,
            margin=dict(l=0,r=5,t=32,b=6),showlegend=False,dragmode='pan',
            font=dict(size=10),hovermode='x')
        st.plotly_chart(figure,use_container_width=True,
                        config={'displaylogo':False,'scrollZoom':True,'responsive':True,'displayModeBar':False})
        st.caption(f'{display_interval} 마지막 봉: {shown.index[-1]} (시세 제공처 지연 가능) · 실시간 체결 보장 없음')
    except Exception as chart_err:
        st.warning(f'{display_interval} 차트를 불러오지 못했습니다: {str(chart_err)[:220]}')
        st.info('무료 시세 공급처가 분봉 조회를 제한할 수 있습니다. 일봉을 선택하거나 다시 조회해 주세요.')
    st.info('아래 목표가·손절가와 AI 상승점수는 **일봉 기반**입니다. 분봉·시간봉으로 바꿔도 분봉용 AI 예측으로 전환되지 않습니다.')
    st.caption('손절·목표가는 최근 확정 스윙 저점/고점·ATR·손익비를 사용한 **롱 포지션 가정의 예시 가격**입니다. 이미 보유한 포지션의 실제 손절 주문은 아니며, 시장 상황에 따라 갱신됩니다. 스윙 추세선은 확정 지점으로만 표시합니다.')
    with st.expander('주요 보조지표 한눈에 보기'):
        st.dataframe(pd.DataFrame(rule_summary(df),columns=['지표','상태']),hide_index=True,use_container_width=True)
    with st.expander('엘리엇 파동 후보 · 확정 스윙 · 피보나치 참고 수준'):
        st.info(f'현재 스윙 구조: **{waves["label"]}**')
        st.caption(waves['notes'])
        if waves['points']:st.dataframe(pd.DataFrame(waves['points']),hide_index=True,use_container_width=True)
        if waves['fib']:st.dataframe(pd.DataFrame([{'되돌림 비율':k,'참고 가격':v} for k,v in waves['fib'].items()]),hide_index=True,use_container_width=True)

if active_section=='forecast':
    st.markdown('#### 과거 검증 기반 가격범위의 실제 포함률')
    interval8=interval_coverage_audit(raw,horizon_days)
    if interval8.get('historical_coverage') is not None:
        first,last=st.columns(2)
        first.metric('선택 기간 후 과거 경험적 범위',
                     f'{price_str(interval8["lower_price"],symbol)} ~ {price_str(interval8["upper_price"],symbol)}')
        last.metric('독립 과거기간 실제 포함률',f'{interval8["historical_coverage"]:.1%}',
                    help=f'총 {interval8["cases"]}건 · 10~90% 범위의 성적')
        st.caption(interval8['notes'])
    else:st.caption('범위 검증: '+interval8['status']+' · 표본 '+str(interval8.get('cases',0))+'건')
    if res.paths:
        paths=res.paths
        st.plotly_chart(forecast_chart(paths,spot,horizon_days),use_container_width=True)
        a,b=st.columns(2)
        c=st.container()
        a.metric('낮은 시나리오 (10%)',price_str(spot*(1+paths['low'][-1]),symbol),f"{paths['low'][-1]*100:+.1f}%")
        b.metric('중앙 시나리오 (50%)',price_str(spot*(1+paths['median'][-1]),symbol),f"{paths['median'][-1]*100:+.1f}%")
        c.metric('높은 시나리오 (90%)',price_str(spot*(1+paths['high'][-1]),symbol),f"{paths['high'][-1]*100:+.1f}%")
        st.caption(f'현재 지표와 유사한 과거 {paths["count"]}개 구간을 찾아 그 다음 {horizon_days}거래일 실제 가격 경로의 분포를 표시합니다. **이는 정식 예측구간이 아니며, 낮은/높은 시나리오가 80% 신뢰구간이라는 뜻도 아닙니다.** AI 상승점수와 경로는 서로 다른 방법을 사용합니다.')
        export=pd.DataFrame({'거래일':range(horizon_days+1),'하위_10퍼센타일':spot*(1+paths['low']),
                             '중앙_50퍼센타일':spot*(1+paths['median']), '상위_90퍼센타일':spot*(1+paths['high'])})
        st.download_button(f'{horizon_days}일 경로 CSV 내려받기',data=export.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'{symbol}_{horizon_days}day_scenarios.csv',mime='text/csv')
    else:
        st.info('과거 유사구간이 부족하여 선택한 기간의 예상 경로를 계산하지 못했습니다.')
    st.markdown('**해석할 때 주의할 점**')
    st.caption('내일 가격을 실시간으로 알거나 선택 기간 후 가격을 확정하는 시스템은 아닙니다. 결과는 과거 시세에서 추출한 통계적 가정이며, 실적 발표·공시·정책·시가 급등락 등의 사건은 반영되지 않을 수 있습니다.')


if active_section=='risk':
    st.subheader('🛡️ 손실 위험 및 자금관리 (매매 주문 아님)')
    st.caption('미래를 예측하는 위험 수치가 아니라 과거 확정 가격으로 계산한 설명용 통계입니다. 폭락·거래정지·갭하락에서는 계획 손절가보다 훨씬 큰 손실이 날 수 있습니다.')
    r1,r2=st.columns(2)
    r3,r4=st.columns(2)
    r1.metric('최근 252일 연율 변동성',f'{risk.ann_volatility:.1%}' if risk.ann_volatility is not None else '—')
    r2.metric('최근 252일 최대 낙폭',f'{risk.max_drawdown:.1%}' if risk.max_drawdown is not None else '—')
    r3.metric('과거 1일 하위 5% 수익률',f'{risk.daily_var95:.1%}' if risk.daily_var95 is not None else '—')
    r4.metric('최악 5%일 평균 수익률',f'{risk.daily_cvar95:.1%}' if risk.daily_cvar95 is not None else '—')
    st.caption(f'최근 하루 최악 종가 변동: {risk.worst_daily_return:.1%}' if risk.worst_daily_return is not None else '하루 최악 변동 계산 표본 부족')
    st.caption(f'과거 최대 시가 갭하락: {risk.worst_open_gap:.1%}' if risk.worst_open_gap is not None else '시가 갭 데이터 부족')
    st.caption(f'최근 20일 중앙 거래대금: {price_str(risk.median_dollar_turnover_20d,symbol)} (통화 및 종목 특성 주의)' if risk.median_dollar_turnover_20d is not None else '거래대금 자료 부족')
    st.markdown('#### 직접 입력하는 가상 손실 예산')
    left,right=st.columns(2)
    with left:
        portfolio=st.number_input('가상의 총 투자금(주가 표시 통화와 동일)',min_value=100.,value=10000.,step=1000.)
        risk_pct=st.slider('한 번의 거래에서 감수할 최대 계획 손실',0.1,5.0,1.0,0.1,format='%.1f%%')
    with right:
        max_alloc=st.slider('이 종목에 넣을 최대 자산 비중',1,100,20,1,format='%d%%')
        fee_pct=st.number_input('예상 왕복 비용(%)',min_value=0.,max_value=5.,value=.3,step=.1)
    try:
        sizing=hypothetical_size(risk.entry_price,risk.stop_price,portfolio,risk_pct,max_alloc)
        st.metric('계획 손실·투자비중 기준의 예시 최대 주식 수',f'{sizing["shares"]:,}주')
        aa,bb=st.columns(2)
        cc=st.container()
        aa.metric('가상 투자금',f'{sizing["position_value"]:,.2f}')
        bb.metric('손절가에 정확히 체결된 경우 손실',f'-{sizing["planned_loss_to_stop"]:,.2f}')
        cc.metric('가정 왕복 비용',f'{sizing["position_value"]*fee_pct/100:,.2f}')
        st.caption(f'진입가 {price_str(risk.entry_price,symbol)} / SL {price_str(risk.stop_price,symbol)} / 1차 목표 {price_str(risk.tp1_price,symbol)} / 목표 손익비 ≈ {risk.reward_risk:.2f}:1' if risk.reward_risk is not None else '손익비 계산 불가')
        st.warning('손절 주문은 지정 가격에 보장 체결되지 않습니다. 갭하락·수수료·세금·환율을 반영하면 실제 손실이 예시보다 클 수 있습니다.')
    except ValueError as exc:
        st.error(str(exc))
    st.markdown('#### 거래비용 변화에 따른 검증 신호 성적')
    stressed=model_cost_stress(strict.rows,fee_pct)
    if stressed['cases']:
        st.caption(f'과거 매수 신호 {stressed["cases"]}건 · 왕복 {fee_pct:.2f}% 비용 가정 시 적중 {stressed["hit_rate"]:.1%} · 평균 순수익 {stressed["avg_net"]:+.2%} · 최악 {stressed["worst"]:+.2%}')
    else:st.caption('거래비용 민감도 분석을 위한 과거 매수 신호가 없습니다.')
    st.caption('실제 bid/ask, 옵션 주문 체결, 슬리피지, 세금 및 갭하락은 포함하지 않는 근사치입니다.')
    st.markdown('#### 기업 실적발표 일정 점검')
    if not is_demo and uploaded is None and st.button('공개 실적 일정 조회',key='earnings_calendar'):
        try:st.session_state['event8']=(symbol,load_earnings_events(symbol))
        except Exception as exc:st.warning('실적 일정 조회 불가: '+str(exc)[:160])
    event8=st.session_state.get('event8')
    if event8 and event8[0]==symbol:
        ev=event8[1]
        if ev.get('known'):
            st.write(f'공개 제공처의 예정 실적일: {ev["date"]} · {ev["days_to_event"]}일 후/전')
            if ev.get('high_risk'):st.warning('7일 이내 실적 이벤트: 갭하락 및 예측 불확실성 확대 가능. 신규 투자 결정에 주의하세요.')
        else:st.info('공개 제공처에서 실적발표일을 확인하지 못했습니다. 실적이 없다는 의미는 아닙니다.')
    st.markdown('#### 과거 기간별 실제 수익률 분포 (겹치는 과거 구간)')
    stress=historical_stress(raw,periods=tuple(sorted({5,21,63,horizon_days})))
    if not stress.empty:
        styled=stress.copy()
        for field in ['최악 과거 수익률','하위 5%','중앙값','상위 95%']:
            styled[field]=styled[field].map(lambda x:f'{x:+.1%}')
        st.dataframe(styled,use_container_width=True,hide_index=True)
        st.caption('이 분포는 독립 예측 검증이 아니라 과거 실제 가격 분포입니다. 위 기간들이 서로 겹치므로 표본 개수를 독립 사건 수로 해석하지 마세요.')

if active_section=='options':
    st.markdown('#### 옵션 관측 기록의 실제 변화')
    archived_options=read_snapshots(CONTEXT_STORE,symbol)
    changes=option_snapshot_changes(archived_options)
    if changes:st.dataframe(pd.DataFrame(changes),hide_index=True,use_container_width=True)
    else:st.caption('실제 관측 기록이 두 번 이상 쌓여야 OI·IV 변화가 나타납니다. 과거 기록을 임의로 만들지 않습니다.')
    st.markdown('#### 옵션 체인 / 미결제약정 / 내재변동성')
    st.caption('미국 상장 옵션을 중심으로 지원합니다. 옵션 데이터는 공급처 사정에 따라 지연·제한될 수 있으며, 시세가 실시간인지 검증되지 않았습니다.')
    if is_demo or uploaded is not None:
        st.info('가상/CSV 가격 데이터에는 옵션 체인을 연결하지 않습니다. 실제 종목을 선택하세요.')
    elif st.button('📥 옵션 데이터 불러오기',key='load_options_button',use_container_width=True):
        try:
            with st.spinner('옵션 체인을 가져와 분석하는 중...'):
                expiry,calls,puts=load_options_chain(symbol)
                opts=analyze_options(spot,calls,puts,expiry)
                st.session_state.options_result=(symbol,date,opts)
        except Exception as e:
            st.session_state.pop('options_result',None)
            st.error(f'옵션 데이터를 가져오지 못했습니다: {type(e).__name__}: {str(e)[:220]}')
    cached_opt=st.session_state.get('options_result')
    opts=(cached_opt[2] if cached_opt and cached_opt[:2]==(symbol,date) else None)
    if opts:
        st.caption(f'선택 만기: {opts.expiry} · 만기까지 {opts.dte}일 · 옵션 조회 당시 기초자산 참고가 ${spot:,.2f}')
        k1,k2=st.columns(2)
        k3,k4=st.columns(2)
        k1.metric('풋/콜 미결제약정 비율',f'{opts.oi_pcr:.2f}' if opts.oi_pcr is not None else '—')
        k2.metric('풋/콜 거래량 비율',f'{opts.volume_pcr:.2f}' if opts.volume_pcr is not None else '—')
        k3.metric('콜 OI 최대 행사가',price_str(opts.call_wall,symbol))
        k4.metric('풋 OI 최대 행사가',price_str(opts.put_wall,symbol))
        l1,l2=st.columns(2)
        l3,l4=st.columns(2)
        l1.metric('Max Pain 참고 행사가',price_str(opts.max_pain,symbol))
        l2.metric('ATM 양매수 가격 합',price_str(opts.atm_straddle,symbol))
        l3.metric('ATM 내재변동성(연율)',f'{opts.implied_vol*100:.1f}%' if opts.implied_vol is not None else '—')
        l4.metric('만기까지 IV 예상 1σ 변동폭',f'±{opts.implied_move_pct*100:.1f}%' if opts.implied_move_pct is not None else '—')
        if opts.atm_straddle is not None:
            st.caption(f'ATM 콜+풋 프리미엄 참고: ±${opts.atm_straddle:,.2f} ({opts.atm_quote_kind}). 이는 수익 보장 범위가 아니며 옵션 가격에는 시간가치가 포함됩니다.')
        if opts.skew_diff is not None:
            st.metric('대략적 OTM 풋IV - 콜IV',f'{opts.skew_diff*100:+.1f}%p',help='풋은 현 주가의 95%, 콜은 105% 근처 행사가. 델타 매칭을 하지 않은 단순 비교입니다.')
        import plotly.express as px
        opt_grp=opts.chain.groupby(['strike','type'],as_index=False)['openInterest'].sum()
        fig_opt=px.bar(opt_grp,x='strike',y='openInterest',color='type',barmode='group',
                        color_discrete_map={'CALL':'#22c55e','PUT':'#ef4444'},
                        title='행사가별 미결제약정(Open Interest)')
        fig_opt.add_vline(x=spot,line_color='#facc15',line_dash='dash')
        fig_opt.update_layout(template='plotly_dark',height=330,margin=dict(l=10,r=10,t=55,b=5))
        st.plotly_chart(fig_opt,use_container_width=True)
        with st.expander('ATM 옵션 이론 그릭스 (델타·감마·세타·베가)'):
            gr=[]
            for label,g in [('콜',opts.call_greeks),('풋',opts.put_greeks)]:
                if g:gr.append({'종류':label,'델타':g['delta'],'감마':g['gamma'],
                                '하루 세타(이론)':g['theta_day'],'IV 1%p당 베가':g['vega_1pct']})
            if gr:st.dataframe(pd.DataFrame(gr),hide_index=True,use_container_width=True)
            else:st.info('내재변동성 데이터가 없어 그릭스를 계산할 수 없습니다.')
        with st.expander('원본 옵션 체인 표'):
            st.dataframe(opts.chain,hide_index=True,use_container_width=True)
        for note in opts.notes:st.caption('• '+note)
        if st.button('📊 만기별 IV 비교 (약 7·30·60일)',key='term_structure'):
            try:
                periods=options_term_structure(symbol,spot)
                if periods.empty:st.info('비교할 수 있는 옵션 만기 데이터가 없습니다.')
                else:st.dataframe(periods,hide_index=True,use_container_width=True)
            except Exception as e:st.warning('만기별 옵션 조회 실패: '+str(e)[:180])
        st.caption('GEX·딜러 감마 노출은 보유 주체·순포지션이 없어 계산하지 않습니다. OI만으로 방향성 감마를 추정하면 오해할 수 있습니다.')
    else:
        st.info('이 종목의 옵션을 분석하려면 위 **옵션 데이터 불러오기** 버튼을 누르세요. 옵션이 없는 종목은 분석하지 않습니다.')

if active_section=='fund':
    st.markdown('#### 회사 재무 + 시장 흐름')
    if is_demo or uploaded is not None:
        st.info('재무/시장 정보는 실제 종목 데이터일 때만 표시합니다.')
    else:
        try:
            fin=load_fundamentals(symbol)
            if fin:
                fdf=pd.DataFrame([{'지표':k,'값':f'{v:,.3f}' if isinstance(v,(float,int)) else str(v)} for k,v in fin.items()])
                st.dataframe(fdf,hide_index=True,use_container_width=True)
                st.caption('재무 수치의 기준 기간 및 단위는 Yahoo 제공 정보에 따릅니다. 회계 기간·통화·단위를 직접 검증하지 않은 수치는 주의하세요.')
            else:st.info('이 종목의 재무 정보가 없습니다.')
        except Exception as e:st.warning(f'재무정보 조회 실패: {str(e)[:180]}')
        try:
            macro=load_macro()
            if macro:
                st.markdown('##### 참고 시장 지표')
                st.dataframe(pd.DataFrame([{'지표':k,'현재값':v['last'],'5거래일 변화(%)':v['five_day_pct'],'기준일':v['asof']} for k,v in macro.items()]),hide_index=True,use_container_width=True)
            else:st.info('시장 참고 데이터를 조회하지 못했습니다.')
        except Exception as e:st.warning(f'시장 데이터 조회 실패: {str(e)[:180]}')
        st.markdown('##### SEC 무료 과거 공시 (미국 상장회사 일부 지원)')
        if not os.environ.get('SEC_USER_AGENT'):
            st.info('SEC 제출일 기준 과거 재무 조회는 SEC_USER_AGENT 등록 후 사용합니다. 이름/연락처가 포함된 User-Agent가 필요하며, 설정하지 않아도 나머지 기능은 작동합니다.')
        elif st.button('📂 SEC 과거 공시 불러오기',key='sec_history_btn'):
            try:
                secs=load_sec_data(symbol)
                st.session_state['sec_result']=(symbol,secs)
            except Exception as e:st.warning('SEC 공시 조회 실패: '+str(e)[:180])
        sec_result=st.session_state.get('sec_result')
        if sec_result and sec_result[0]==symbol:
            filings=sec_result[1]
            if filings.empty:st.info('지원되는 XBRL 재무 공시가 없습니다.')
            else:
                st.dataframe(filings.tail(30),hide_index=True,use_container_width=True)
                sec_features=financial_features_asof(raw.index,filings)
                st.line_chart(sec_features[['sec_revenue_yoy']].dropna())
                st.caption('미래 공시를 과거 가격에 붙이지 않도록 공시 제출일 다음 거래일부터 반영합니다. 연간 보고서 수치와 일부 미국 GAAP 태그만 비교하며, 회계년도 차이·재공시 여부는 따로 확인해야 합니다.')
        st.caption('시장 지표는 해당 종목의 단기 방향을 보장하지 않습니다. 분단위 호가·옵션 주문흐름·다크풀 데이터는 포함되지 않습니다.')
        st.divider()
        st.subheader('📐 비교지수 대비 상대 성과')
        bm=st.selectbox('비교 대상 지수/ETF',['SPY','QQQ','SMH','^KS11','^KQ11'],index=3 if symbol.endswith(('.KS','.KQ')) else 0)
        if st.button('비교지수와 수익률·베타 비교',key='compare_market'):
            try:
                src=load_benchmark(bm)
                meta,table,normalized=compare_market(raw,src)
                st.session_state['benchmark_results']=(symbol,bm,date,meta,table,normalized)
            except Exception as e:st.warning('시장 비교 실패: '+str(e)[:200])
        market_comp=st.session_state.get('benchmark_results')
        if market_comp and market_comp[:3]==(symbol,bm,date):
            meta,table,normalized=market_comp[3:]
            x1,x2=st.columns(2)
            x1.metric('최근 252거래일 베타',f'{meta["beta"]:.2f}' if meta['beta'] is not None else '—')
            x2.metric('최근 252거래일 상관계수',f'{meta["correlation"]:.2f}' if meta['correlation'] is not None else '—')
            st.line_chart(normalized)
            if not table.empty:
                tidy=table.copy()
                for col in ['종목','비교지수','초과 수익률']:
                    tidy[col]=tidy[col].map(lambda v:f'{v:+.1%}')
                st.dataframe(tidy,hide_index=True,use_container_width=True)
            st.caption('비교지수와 날짜가 겹치는 종가를 100으로 맞췄습니다. 베타·상관관계는 과거 수치이며 미래 안정성이나 투자 성과를 보장하지 않습니다.')

if active_section=='news':
    if uploaded is not None or is_demo:
        st.info('CSV·가상 데이터에는 종목 뉴스가 연결되지 않습니다.')
    else:
        st.caption('최근 관련 기사 제목과 원문 링크만 제공합니다. **현재 학습 모델은 뉴스 내용을 읽거나 뉴스 감성을 학습하지 않습니다.**')
        news=latest_news(symbol)
        usable=[readable_news(x) for x in news]
        usable=[x for x in usable if x and x[0]]
        if not usable:st.info('뉴스 공급처에서 관련 기사를 반환하지 않았습니다.')
        if usable:
            st.markdown('##### 무료 제목 단어 분석 (참고용)')
            st.dataframe(pd.DataFrame(news_headline_score([x[0] for x in usable])),hide_index=True,use_container_width=True)
            st.caption('단순 영어 단어 사전 분석입니다. 기사 본문, 부정 표현, 문맥을 이해하지 못하므로 AI 예측 신호로 사용하지 않습니다.')
        for title,url,publisher in usable:
            if isinstance(url,str) and url.startswith('http'):
                st.markdown(f'- [{title}]({url})' + (f' — {publisher}' if publisher else ''))
            else:st.markdown(f'- {title}' + (f' — {publisher}' if publisher else ''))
        st.link_button('Yahoo Finance에서 뉴스 직접 확인',f'https://finance.yahoo.com/quote/{symbol}/news/')

if active_section=='explain':
    st.subheader('비용 없는 종합 리서치 리포트')
    st.caption('실제 확보한 기술지표·옵션·뉴스 제목·기업정보만 연결합니다. 규칙 기반 보고서이며 외부 생성형 AI가 아닙니다.')
    if is_demo or uploaded is not None:
        st.info('가상/CSV 데이터는 외부 옵션·뉴스·기업 재무를 자동 조회하지 않습니다.')
    if st.button('🧩 무료 전체 분석 데이터 조회 및 리포트 생성',use_container_width=True,key='full_report'):
        news_rows=[]; fin={}; market={}; option=None
        if not is_demo and uploaded is None:
            try:
                news_rows=news_headline_score([i[0] for item in latest_news(symbol) if (i:=readable_news(item)) and i[0]])
            except Exception as e:st.warning('뉴스 제공처 연결 실패: '+str(e)[:110])
            try:fin=load_fundamentals(symbol)
            except Exception as e:st.warning('재무 데이터 연결 실패: '+str(e)[:110])
            try:market=load_macro()
            except Exception as e:st.warning('시장 데이터 연결 실패: '+str(e)[:110])
            opt_cache=st.session_state.get('options_result')
            if opt_cache and opt_cache[:2]==(symbol,date):option=opt_cache[2]
            elif not symbol.endswith(('.KS','.KQ')):
                try:
                    ex,ca,pu=load_options_chain(symbol)
                    option=analyze_options(spot,ca,pu,ex)
                    st.session_state.options_result=(symbol,date,option)
                except Exception as e:st.caption('옵션 미연결: '+str(e)[:110])
        text_result=free_explanation(symbol,df,res,adapted,news_rows,option,fin,market,horizon_days=horizon_days)
        st.session_state['free_report']=(symbol,date,horizon_days,text_result,
            build_llm_context(symbol,spot,
                {'model_probability_uncalibrated':res.probability, 'selected_horizon_days':horizon_days, 'strict_audit_decision':strict.recommendation,'data_quality_status':health.status,'model_validation_cases':res.test_cases,
                 'adaptive_score_not_probability':adapted.score,'trend_regime':adapted.regime,
                 'wave_structure_candidate':waves['label']},
                res.levels,option,fin,[n['headline'] for n in news_rows],market))
    saved=st.session_state.get('free_report')
    if saved and saved[:3]==(symbol,date,horizon_days):
        st.markdown(saved[3])
        st.download_button('리포트 텍스트 내려받기',saved[3].encode('utf-8'),
                           file_name=f'{symbol}_free_report.md')
    st.divider()
    st.subheader('PC용 무료 로컬 생성형 AI (선택)')
    st.caption('PC에 Ollama를 설치하고 모델을 내려받아 실행해야 합니다. 모바일 단독 또는 무료 클라우드에는 해당 AI가 자동 설치되지 않습니다. 과거 옵션/뉴스는 가격 머신러닝의 학습 특성이 아닙니다.')
    model_name=st.text_input('로컬 모델 이름',value='gemma3:4b',help='Ollama에 설치된 모델 이름을 입력하세요.')
    if st.button('🧠 내 PC 로컬 AI에게 보고서 자세히 해설받기',key='ollama_button'):
        report=st.session_state.get('free_report')
        if not report or report[:3]!=(symbol,date,horizon_days):st.warning('먼저 무료 전체 분석 데이터를 조회해 주세요.')
        else:
            try:
                response=explain_with_ollama(report[4],model=model_name)
                st.session_state['local_answer']=(symbol,date,horizon_days,response)
            except Exception as e:
                st.warning('로컬 AI가 실행 중이지 않거나 연결할 수 없습니다: '+str(e)[:180])
    ans=st.session_state.get('local_answer')
    if ans and ans[:3]==(symbol,date,horizon_days):st.markdown(ans[3])

if active_section=='model':
    st.markdown('#### 모델 점수 신뢰도 · 시간순 확률 보정 (8.0)')
    calibrated=reliability_calibration(strict.rows,res.probability)
    st.write(calibrated['status'])
    if 'raw_brier' in calibrated:
        st.caption(f'시간순 앞부분 {calibrated["train"]}건으로 보정하고 뒷부분 {calibrated["test"]}건 검증: 원본 Brier {calibrated["raw_brier"]:.3f} / 보정 Brier {calibrated["cal_brier"]:.3f} / 과거 기준선 {calibrated["prior_brier"]:.3f}')
    if calibrated.get('calibrated_latest') is not None:
        st.metric('보정된 참고 상승빈도 추정 (실제 확률 아님)',f'{calibrated["calibrated_latest"]:.1%}')
    st.caption('후반 검증에서도 성능이 개선되었을 때만 참고 보정값을 제공합니다. 과거의 점수와 결과로 적합한 단순 단조 보정으로, 미래 보장이나 매수 신호가 아닙니다.')
    st.markdown('#### 과거 잘못된 분석의 공통 상황')
    errors8=forecast_error_patterns(strict.rows)
    if errors8.get('groups'):
        failures8=pd.DataFrame(errors8['groups'])
        failures8['오분류율']=failures8['오분류율'].map(lambda x:f'{x:.1%}')
        st.dataframe(failures8,hide_index=True,use_container_width=True)
        st.caption('예측 실패가 특정 시장 조건에 많이 발생했는지의 연관성만 봅니다. 뉴스·실적 때문이라고 임의로 단정하지 않습니다.')
    else:st.caption(errors8['status'])
    st.markdown('#### 시장 국면별 모델 확인')
    st.write('현재 추정 상태: **'+regime8['regime']+'**')
    st.caption('국면이 달라지면 무조건 동일한 전략을 적용하지 않습니다. 아직 국면마다 별도의 AI 신경망을 완전 자동 교체하는 기능은 아니며, 과거 국면별 검증 결과를 제공합니다.')
    if st.button('🧪 신규 지표 후보 독립기간 테스트',key='lab8'):
        with st.spinner('5가지 지표 규칙의 앞기간·미사용 뒷기간 성과 확인 중...'):
            st.session_state['lab8_result']=(symbol,horizon_days,indicator_candidate_lab(raw,horizon_days))
    lab=st.session_state.get('lab8_result')
    if lab and lab[:2]==(symbol,horizon_days):
        st.caption(lab[2]['status'])
        if lab[2]['rows']:st.dataframe(pd.DataFrame(lab[2]['rows']),use_container_width=True,hide_index=True)
        st.caption(lab[2].get('note','검증 표본 부족. 자동 채택하지 않습니다.'))
    st.divider()
    st.markdown('#### 무료 과거 뉴스·공시 검색 + 과거 시점 학습 (7.0)')
    if is_demo or uploaded is not None:
        st.info('가상/CSV 데이터는 외부 옵션·뉴스 기록과 결합하지 않습니다.')
    else:
        st.caption('GDELT 무료 과거 뉴스 제목과 SEC 과거 공시를 검색합니다. 공개된 자료만 사용하며 과거 날짜를 마음대로 조작하지 않습니다. 과거 옵션 IV·OI 전체 기록은 무료로 자동 복원할 수 없습니다.')
        company_default= {'CRWV':'CoreWeave','NVDA':'NVIDIA','TSLA':'Tesla','AAPL':'Apple',
                          'MSFT':'Microsoft','GOOGL':'Alphabet','META':'Meta Platforms',
                          'AMZN':'Amazon','AMD':'Advanced Micro Devices',
                          'PLTR':'Palantir','SOFI':'SoFi','COIN':'Coinbase'}.get(symbol,symbol)
        historical_company=st.text_input('과거 뉴스에서 검색할 정확한 회사 영문명',value=company_default,key='history_company')
        months=st.selectbox('검색할 과거 범위', [3,6,12,24],index=1,format_func=lambda x:f'최근 {x}개월')
        if st.button('🔎 무료 과거 뉴스 검색해서 학습자료 채우기',key='history_news_btn'):
            try:
                with st.spinner('공개된 과거 뉴스 검색 중. 자료 제공처에서 제한하면 불완전 결과를 표시합니다.'):
                    start=(pd.Timestamp.now(tz='UTC')-pd.DateOffset(months=months)).date()
                    end=pd.Timestamp.now(tz='UTC').date()
                    result=backfill_news(symbol,historical_company,start,end,path=CONTEXT_STORE,
                                         request_budget=35,chunk_days=7,throttle_seconds=.6)
                st.session_state['news_backfill_result']=result
                do_context_assessment.clear()
            except Exception as ex:st.error(f'과거 뉴스 자료 요청 실패: {str(ex)[:200]}')
        result=st.session_state.get('news_backfill_result')
        if result and result['symbol']==symbol:
            st.write(f'GDELT 과거 뉴스 날짜 {result["saved_days"]}일 새로 보관 / 총 {result["coverage_days"]}일')
            if result['partial']:
                st.warning('일부 기간의 검색이 완료되지 않았습니다. 나중에 다시 실행하면 중복 날짜를 건너뜁니다.')
                if result['errors']:st.caption('검색 오류 예: '+result['errors'][0])
        if st.button('🏛️ SEC 과거 재무 공시 무료 검색·저장',key='sec_backfill_btn'):
            try:
                with st.spinner('SEC 실제 과거 제출일로 재무자료 가져오는 중'):
                    downloaded=save_sec_filings(symbol)
                st.success(f'SEC 과거 공시 {downloaded["filings"]}건 저장')
                do_context_assessment.clear()
            except Exception as ex:
                st.warning('SEC 공시 조회 실패: '+str(ex)[:230]+' (SEC_USER_AGENT 등록이 필요할 수 있습니다.)')
        archive=read_snapshots(CONTEXT_STORE,symbol)
        sec_cache=st.session_state.get('sec_result')
        filings=(sec_cache[1] if sec_cache and sec_cache[0]==symbol else load_sec_filings(symbol))
        sec_feats=financial_features_asof(raw.index,filings) if not filings.empty else None
        ctx=do_context_assessment(raw,archive,horizon_days,sec_feats)
        st.write('**학습 상태:** '+ctx.status)
        st.caption(ctx.explanation)
        if ctx.included_features:st.caption('실제로 사용한 기록 변수: '+', '.join(ctx.included_features))
        a,b=st.columns(2)
        c=st.container()
        a.metric('옵션 관측 가능 일봉',ctx.coverage['option_coverage'])
        b.metric('뉴스 관측 가능 일봉',ctx.coverage['news_coverage'])
        c.metric('독립 보조 모델 검증',ctx.holdout_cases)
        if ctx.score is not None:
            st.metric('검증된 외부자료 보조 모델 상승점수(미보정)',f'{ctx.score:.1%}')
        if ctx.brier_context is not None:
            st.caption(f'보조 모델 Brier {ctx.brier_context:.3f} / 동일 시기의 가격 모델 {ctx.brier_price_only:.3f} / 당시 과거 상승빈도 {ctx.brier_prior:.3f} — 작을수록 좋음')
        st.caption('공개 과거 뉴스·SEC 공시는 수집 직후 날짜를 검증해 학습에 이용할 수 있습니다. 옵션 체인은 과거 전부가 무료로 공개되지 않으므로 실제 기록만 사용합니다. 검증 통과 전에는 판단 보류합니다.')
        if not archive.empty:
            with st.expander('저장된 실제 옵션·뉴스 관측 이력'):
                st.dataframe(archive.tail(40),hide_index=True,use_container_width=True)
        else:
            st.info('저장된 날짜별 자료가 없습니다. GitHub Actions의 정기 수집을 시작하면 이력이 누적됩니다.')
    st.markdown('#### 모델은 무엇을 학습하나요?')
    st.subheader('실전 과거 검증: 워크포워드 (뒤 날짜 정보 사용 금지)')
    st.caption('과거 확률점수의 Brier 오차: 모델 '+(f'{strict.score_brier:.3f}' if strict.score_brier is not None else '—')+' / 당시 알려진 상승빈도 기준선 '+(f'{strict.baseline_brier:.3f}' if strict.baseline_brier is not None else '—')+' (작을수록 좋음)')
    if strict.regime_reports:
        st.markdown('##### 고변동·저변동 시장에서의 차이')
        st.dataframe(pd.DataFrame(strict.regime_reports),hide_index=True,use_container_width=True)
        st.caption('각 평가 날짜에 **당시** 관측 가능했던 과거 변동성으로 분류합니다. 소표본 국면 성적은 신뢰할 수 없습니다.')
    if strict.reliability_bins:
        st.markdown('##### 과거 점수와 실제 상승 빈도 비교')
        st.dataframe(pd.DataFrame(strict.reliability_bins),hide_index=True,use_container_width=True)
        st.caption('각 구간 독립 표본이 5건 미만이면 상승 빈도를 숨깁니다. 이것은 확률 보정 모델이 아닙니다.')
    st.caption('매 검증 날짜마다 해당 날짜까지 공개된 가격으로 지표를 만들고, 결과가 확정된 과거 정답만 학습합니다. 미래 차트로 정답을 외우지 않습니다.')
    if strict.rows:
        st.dataframe(pd.DataFrame(strict.rows),hide_index=True,use_container_width=True)
        st.caption(f'매수 검토 신호 {strict.evaluated_long_signals}건 / 시기별 검증 일관성: {"충족" if strict.positive_halves else "미충족"} / 매수 신호 과거 평균 순수익률: {strict.long_avg_net_return:.2%}' if strict.long_avg_net_return is not None else '매수 신호 발생 이력이 없습니다.')
    st.caption('강한 신호 임계값은 검증하기 전에 고정했습니다. 과거 검증 결과를 보고 유리한 기준을 역으로 선택하지 않습니다. 95% 하한은 확률에 대한 보장이 아니며, 실거래 미끄러짐·세금·장중 변동을 전부 재현하지 못합니다.')
    st.subheader('성과에 따라 바뀌는 보조지표 가중치')
    st.caption(adapted.explanation)
    if adapted.rows:
        wdf=pd.DataFrame(adapted.rows)
        st.dataframe(wdf,hide_index=True,use_container_width=True,
            column_config={'과거 방향 적중률':st.column_config.NumberColumn(format='%.3f',help='0~1의 비율 값입니다.')})
        st.caption(f'적중률은 후보 규칙의 비중복 {horizon_days}거래일 방향 일치율이고 비용·변동성·실거래 손익을 반영하지 않습니다. 최종 학습 모델 확률과 별개의 투표 점수입니다.')
    st.write(f'최근 10년까지의 확정 일봉에서 모멘텀, 이동평균, RSI, 스토캐스틱 3개, MACD, ADX, ATR, 밴드 및 거래량 특징을 계산합니다. 학습 목표는 **{horizon_days}거래일 후 종가가 현재보다 높았는지**입니다.')
    st.write('매 분석 시 과거의 결과가 확정된 거래일만 사용해 랜덤 포레스트를 다시 학습합니다. 두 번째 학습 기능은 과거 성과에 따라 8개 규칙의 가중치를 다시 계산합니다. 스스로 새로운 알고리즘 코드를 작성하지는 않습니다.')
    if res.probability is not None:
        aa,bb=st.columns(2)
        cc=st.container()
        aa.metric('학습 사례(겹치는 구간 포함)',f'{res.train_cases:,}개')
        bb.metric(f'독립 검증 건수(약 {horizon_days}일 간격)',f'{res.test_cases}개')
        cc.metric('Brier 점수(낮을수록 좋음)',f'{res.backtest_brier:.3f}' if res.backtest_brier is not None else '—')
        st.caption(f'검증은 과거 구간을 시간순으로 분리하고, {horizon_days}거래일의 미래 수익률이 평가 구간에 침범하지 않도록 경계를 비웁니다. 다만 하나의 최근 검증 구간만 사용하므로 성능은 불안정할 수 있습니다.')
        importance=pd.DataFrame([{'기술지표':FEATURE_LABELS.get(k,k),'중요도':round(float(v),3)} for k,v in res.model_features])
        st.bar_chart(importance.set_index('기술지표'))
        st.caption('중요도는 모델이 학습 중 자주 사용한 변수입니다. 주가를 특정 방향으로 움직이게 했다는 인과관계는 아닙니다.')
        if not is_demo and uploaded is None:
            recorded,hit=persist_prediction(symbol,raw,res.probability,horizon_days)
            if recorded:
                st.metric('이전 방문 예측의 실제 적중률',f'{hit*100:.1f}%',f'{recorded}건 완료')
            else:
                st.caption(f'오늘부터 같은 종목의 예측을 기록합니다. {horizon_days}거래일이 지난 뒤 다시 분석하면 실제 방향과 비교합니다.')
        st.caption('방문별 SQLite 기록은 무료 클라우드 재배포 시 사라질 수 있습니다. 지속 기록은 GitHub Actions가 커밋하는 data/predictions.csv 및 data/context_snapshots.jsonl을 기준으로 합니다.')
    st.markdown('##### 자동 재학습 기록 (GitHub 정기 실행 시 저장)')
    if JOURNAL.exists():
        try:
            hist=pd.read_csv(JOURNAL)
            relevant=hist[(hist.symbol.astype(str)==symbol) & (pd.to_numeric(hist.get('horizon_days',pd.Series(21,index=hist.index)),errors='coerce')==horizon_days)]
            if len(relevant):
                observed=relevant[relevant.direction_correct.notna()]
                st.caption(f'과거 예측 {len(relevant)}건 · 결과 확정 {len(observed)}건 · 결과별 적중률 '+
                    (f'{observed.direction_correct.mean():.1%}' if len(observed) else '아직 없음'))
                st.dataframe(relevant.tail(20),hide_index=True,use_container_width=True)
            else:st.caption('아직 이 종목에 저장된 예측 기록이 없습니다.')
        except Exception as e:st.caption('자동 예측 기록을 읽지 못했습니다: '+str(e)[:150])
    else:
        st.caption('처음에는 기록이 없습니다. GitHub Actions가 한 번 실행되면 날짜별 기록이 생성됩니다.')
    latest_report=Path(__file__).with_name('reports')/'latest.md'
    if latest_report.exists():
        with st.expander('마지막 자동 리서치 실행 결과'):
            st.markdown(latest_report.read_text(encoding='utf-8'))
    st.divider()
    st.warning('**투자 유의:** 숫자 형태의 상승확률과 적중률은 모델의 미보정 추정치입니다. 실제로 그 비율대로 맞을 것으로 보장되지 않습니다. 투자 판단과 손실 책임은 사용자에게 있으며 실거래 전 충분한 독립 검증이 필요합니다.')

if active_section=='quick':
    st.markdown('#### 관심종목 여러 개 한눈에 비교')
    st.caption('3개월 실제 수익률, 20일 변동성, 60일선 이격도를 비교합니다. 이 순위가 AI의 매수 추천 순위는 아닙니다.')
    tracked=st.multiselect('비교할 종목 (최대 5개)',options=list(POPULAR),default=['NVDA','CRWV','AAPL'],
                            max_selections=5,format_func=lambda t:f'{POPULAR[t]} ({t})')
    if st.button('📋 선택한 주식 비교',key='multi_screen'):
        result={};errors=[]
        if is_demo or uploaded is not None:st.warning('실제 종목 시세 연결에서만 비교할 수 있습니다.')
        else:
            for sym in tracked:
                try:result[sym]=load_screener_prices(sym)
                except Exception as ex:errors.append(f'{sym}: {type(ex).__name__}')
            st.session_state['multi_screen']=(tracked,watchlist_relative(result),errors)
    comparison=st.session_state.get('multi_screen')
    if comparison and comparison[0]==tracked:
        if comparison[1]:
            table=pd.DataFrame(comparison[1])
            shown_table=table.copy()
            for col in ['3개월 수익률','최근20일 변동성(연율)','60일 이평 대비']:
                shown_table[col]=shown_table[col].map(lambda x:f'{x*100:+.2f}%' if pd.notna(x) else '—')
            st.dataframe(shown_table,hide_index=True,use_container_width=True)
            st.caption('화면은 퍼센트, 내려받는 CSV는 소수 비율입니다. 실제 거래비용은 미반영입니다.')
            st.download_button('비교 결과 CSV 다운로드',table.to_csv(index=False).encode('utf-8-sig'),
                               file_name='smartstock_watchlist_comparison.csv',mime='text/csv')
        for err in comparison[2]:st.warning('조회 실패: '+err)
    st.divider()
    st.markdown('#### 모바일 조건 알림 (선택 기능)')
    st.caption('프로그램 화면은 새로고침 시 분석하고, GitHub 예약 실행에서는 조건 변경 시 Telegram 메시지를 발송할 수 있습니다. 자동 알림 사용에는 개인 Bot Token과 Chat ID를 GitHub Secrets에 별도 등록해야 합니다.')
    st.info('자동 알림은 앱이 아닌 GitHub Actions의 정기 실행에 의존하므로 실시간 알림이 아니며, 설정하지 않으면 완전히 꺼져 있습니다.')

st.caption('SmartStock AI FREE 9.0 모바일 · 무료 API 데이터 조회·기술적 머신러닝·옵션 스냅샷·가격 규칙 가중치 학습. 투자자문/주문 기능 없음.')
