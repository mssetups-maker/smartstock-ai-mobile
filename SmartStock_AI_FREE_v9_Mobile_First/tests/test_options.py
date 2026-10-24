import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import date
import pandas as pd
from options_engine import analyze_options,black_scholes_greeks,pick_expiry
from intelligence import news_headline_score, fundamentals_snapshot, build_llm_context


def sample_chain():
    calls=pd.DataFrame({'strike':[90,100,110], 'openInterest':[90,300,160],
      'volume':[23,25,11],'impliedVolatility':[.29,.30,.33],
      'bid':[11,5,1],'ask':[12,5.4,1.3],'lastPrice':[11.5,5.2,1.2]})
    puts=pd.DataFrame({'strike':[90,100,110],'openInterest':[200,120,50],
      'volume':[40,30,12],'impliedVolatility':[.36,.34,.31],
      'bid':[1.2,4,10],'ask':[1.4,4.4,11],'lastPrice':[1.3,4.2,10.5]})
    return calls,puts


def test_option_metrics_and_price_math():
    c,p=sample_chain()
    o=analyze_options(100,c,p,'2026-11-20',reference_date=date(2026,10,23))
    assert o.dte==28
    assert o.call_oi==550 and o.put_oi==370
    assert o.call_volume==59 and o.put_volume==82
    assert abs(o.oi_pcr-370/550)<1e-10
    assert o.call_wall==100 and o.put_wall==90
    assert o.max_pain in (90,100,110)
    assert abs(o.atm_straddle-9.4)<1e-10
    assert o.implied_move_pct>0 and o.implied_vol>0
    assert o.call_greeks['delta']>0 and o.put_greeks['delta']<0
    assert o.call_greeks['gamma']>0 and o.put_greeks['gamma']>0  # call/put IV differs


def test_invalid_quotes_and_missing_oi_are_safe():
    c,p=sample_chain()
    c['bid']=-1;c['ask']=0;c['lastPrice']=0
    c['openInterest']=0
    o=analyze_options(100,c,p,'2026-11-20',reference_date=date(2026,10,23))
    assert o.oi_pcr is None
    assert o.atm_straddle is None
    assert o.call_wall is None


def test_expiry_and_text_analytics():
    assert pick_expiry(['invalid','2026-11-27','2026-11-10'],date(2026,10,23))=='2026-11-27'
    n=news_headline_score(['Sales growth and record profit','Major fraud probe','Neutral headline'])
    assert n[0]['score']>0 and n[1]['score']<0 and n[2]['score']==0
    assert fundamentals_snapshot({'trailingPE':23,'bad':999})=={'과거 PER':23}
    assert 'last_close' in build_llm_context('ABC',100,{}, {},None,{},['Headline'])


def test_greeks_no_zero_vol():
    assert black_scholes_greeks(100,100,0,20,'call') is None
