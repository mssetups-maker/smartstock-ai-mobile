"""Optional SEC EDGAR point-in-time US GAAP financial series.

Set SEC_USER_AGENT='Personal Research contact@example.com' yourself, per SEC
identification policy. The current snapshot alone is NEVER backdated.
Annual flow tags are limited to ~full fiscal-year observations to avoid mixing
quarterly and annual values. Financial filings are available only AFTER filed.
"""
from __future__ import annotations
import os
from datetime import date
from pathlib import Path
import pandas as pd
import requests

TAGS={
 'revenue_usd':['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet'],
 'net_income_usd':['NetIncomeLoss'],
 'assets_usd':['Assets'],
 'debt_usd':['LongTermDebt','LongTermDebtNoncurrent'],
}
FLOW={'revenue_usd','net_income_usd'}


def extract_filing_series(companyfacts:dict)->pd.DataFrame:
    """Keep filing-date and period-end; exclude filings not identifiable to SEC forms."""
    us=(companyfacts.get('facts') or {}).get('us-gaap') or {}
    rows=[]
    for name,tags in TAGS.items():
        for tag in tags:
            data=(us.get(tag) or {}).get('units',{}).get('USD') or []
            if not data:continue
            for item in data:
                if item.get('form') not in ('10-Q','10-K','10-Q/A','10-K/A'):continue
                if not item.get('filed') or not item.get('end'):continue
                try:
                    filed=pd.Timestamp(item['filed']).normalize()
                    period=pd.Timestamp(item['end']).normalize()
                    value=float(item['val'])
                    if name in FLOW:
                        if not item.get('start'):continue
                        days=(period-pd.Timestamp(item['start'])).days
                        if not 330<=days<=380:continue
                    # End must be before the report was actually filed.
                    if filed<period:continue
                    rows.append({'name':name,'filed':filed,'end':period,'value':value,'tag':tag,'form':item['form']})
                except (ValueError,TypeError,KeyError):continue
            # Use first supported tag with data to avoid duplicating aliases.
            if any(x['name']==name for x in rows):break
    if not rows:return pd.DataFrame(columns=['name','filed','end','value','tag','form'])
    return pd.DataFrame(rows).drop_duplicates(['name','filed','end','value']).sort_values(['filed','end'])


def financial_features_asof(index,filings:pd.DataFrame):
    """No future filings: each bar uses only filings with filed DATE < bar DATE."""
    idx=pd.DatetimeIndex(index).tz_localize(None).normalize()
    result=pd.DataFrame(index=pd.DatetimeIndex(index),columns=['sec_revenue_yoy','sec_roa','sec_debt_assets'],dtype=float)
    if filings is None or filings.empty:return result
    for pos,day in enumerate(idx):
        subset=filings[filings.filed<day]
        if subset.empty:continue
        values={}
        for name in TAGS:
            group=subset[subset.name==name]
            if group.empty:continue
            # Most recent period-end that was actually public, latest amendment by filed date.
            group=group.sort_values(['end','filed']).drop_duplicates(['end'],keep='last')
            latest=group.iloc[-1]
            if (day-latest.filed).days>540:continue
            values[name]=float(latest.value)
            if name=='revenue_usd' and len(group)>=2:
                previous=group.iloc[-2]
                if 300<=(latest.end-previous.end).days<=450 and previous.value>0:
                    result.iat[pos,0]=float(latest.value/previous.value-1)
        if values.get('assets_usd',0)>0:
            if 'net_income_usd' in values:result.iat[pos,1]=values['net_income_usd']/values['assets_usd']
            if 'debt_usd' in values:result.iat[pos,2]=values['debt_usd']/values['assets_usd']
    return result


def get_companyfacts(symbol:str, user_agent:str|None=None, session=None)->dict:
    agent=(user_agent or os.environ.get('SEC_USER_AGENT','')).strip()
    if not agent or '@' not in agent:
        raise ValueError('SEC_USER_AGENT 환경변수에 본인 연락처를 등록해야 SEC 무료 공시를 조회할 수 있습니다.')
    sess=session or requests.Session()
    headers={'User-Agent':agent,'Accept-Encoding':'gzip, deflate','Accept':'application/json'}
    company=sess.get('https://www.sec.gov/files/company_tickers.json',headers=headers,timeout=20)
    company.raise_for_status()
    symbol=symbol.upper().replace('.','-')
    entry=next((v for v in company.json().values() if str(v.get('ticker','')).upper()==symbol),None)
    if entry is None:raise LookupError('SEC 상장 티커 매핑 없음 (미국 기업이 아니거나 이름이 다를 수 있음)')
    cik=f"{int(entry['cik_str']):010d}"
    r=sess.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',headers=headers,timeout=25)
    r.raise_for_status()
    return r.json()
