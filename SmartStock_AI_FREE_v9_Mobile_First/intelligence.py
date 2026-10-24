"""Data-aware fundamental, context and optional LLM interpretation utilities."""
from __future__ import annotations
import json
import math
import pandas as pd

KEYS={
 'shortName':'회사명','sector':'섹터','industry':'산업',
 'marketCap':'시가총액','trailingPE':'과거 PER','forwardPE':'예상 PER',
 'priceToSalesTrailing12Months':'매출액 대비 시가총액(P/S)',
 'revenueGrowth':'매출 성장률','earningsGrowth':'순이익 성장률',
 'profitMargins':'순이익률','operatingMargins':'영업이익률',
 'returnOnEquity':'자기자본이익률(ROE)','totalDebt':'총부채',
 'totalCash':'현금·현금성자산','freeCashflow':'잉여현금흐름',
 'shortPercentOfFloat':'유통주식 대비 공매도 잔고',
 'sharesShort':'공매도 잔고 주식수','heldPercentInstitutions':'기관 지분율',
 'recommendationMean':'애널리스트 권고 평균(낮을수록 매수)',
 'targetMeanPrice':'애널리스트 평균 목표주가',
 'beta':'시장 베타'
}


def fundamentals_snapshot(info):
    if not isinstance(info,dict):return {}
    result={}
    for k in KEYS:
        x=info.get(k)
        if isinstance(x,(int,float)):
            if not math.isfinite(x):continue
            result[KEYS[k]]=x
        elif isinstance(x,str) and x.strip():result[KEYS[k]]=x
    return result


def news_headline_score(titles:list[str]):
    """Simple dictionary-based English headline sentiment, NOT a trained predictive model."""
    positive={'beat','beats','surge','growth','profit','upgrade','strong','record','raise','rally','expand','contract','approval','gain','rebound','partnership','outperform','bullish'}
    negative={'miss','misses','plunge','decline','loss','downgrade','weak','probe','lawsuit','cut','crash','warning','fraud','drop','layoff','bearish','default'}
    results=[]
    for title in titles:
        toks={x.strip('.,:!?()[]').lower() for x in str(title).split()}
        score=len(toks & positive)-len(toks & negative)
        results.append({'headline':str(title),'score':score,'tag':'긍정 단어 우세' if score>0 else '부정 단어 우세' if score<0 else '판단 불가'})
    return results


def build_llm_context(symbol,spot,technical,levels,options,financial,headlines,macro=None):
    """Serialize only observed inputs to LLM, explicitly noting missingness."""
    options_brief=None
    if options:
        options_brief={k:getattr(options,k,None) for k in ['expiry','oi_pcr','volume_pcr','max_pain','call_wall','put_wall','atm_straddle','implied_move_pct','skew_diff']}
    ctx={'symbol':symbol,'last_close':spot,'technical':technical,'levels':levels,
         'options':options_brief,'financial':financial,
         'news_headlines_only':headlines[:12], 'macro':macro or {},
         'limitations':f'last confirmed daily closing price; sources can be delayed; no intraday real-time guarantee; no full article text; horizon {technical.get("selected_horizon_days", "selected")} sessions; model output uncalibrated'}
    return json.dumps(ctx,ensure_ascii=False,default=str)
