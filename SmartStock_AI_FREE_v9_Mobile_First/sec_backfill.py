"""Cache genuine historic SEC filing facts for as-of-date reuse.

The companyfacts API requires an identifying contact User-Agent under SEC
fair-access policy; the program never creates a fake contact address.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sec_history import get_companyfacts, extract_filing_series
from point_in_time import safe_symbol

ROOT=Path(__file__).resolve().parent

def sec_cache_path(symbol,root=ROOT):
    return Path(root)/'data'/'sec_facts'/f'{safe_symbol(symbol)}.csv'

def save_sec_filings(symbol, *, root=ROOT, user_agent=None, session=None):
    facts=get_companyfacts(symbol,user_agent=user_agent,session=session)
    filings=extract_filing_series(facts)
    if filings.empty:return {'symbol':symbol,'filings':0,'path':None}
    path=sec_cache_path(symbol,root)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Underlying end/filed dates are preserved, not replaced with download date.
    filings.to_csv(path,index=False)
    return {'symbol':symbol,'filings':len(filings),'path':str(path)}

def load_sec_filings(symbol,root=ROOT):
    path=sec_cache_path(symbol,root)
    if not path.exists():
        return pd.DataFrame(columns=['name','filed','end','value','tag','form'])
    data=pd.read_csv(path,parse_dates=['filed','end'])
    expected={'name','filed','end','value','tag','form'}
    if not expected.issubset(data.columns):raise ValueError('SEC 캐시 열 구성이 올바르지 않습니다')
    return data.sort_values(['filed','end'])
