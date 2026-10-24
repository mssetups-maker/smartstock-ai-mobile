"""CSV prediction/outcome ledger, separated by selected horizon."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from engine import HORIZON, validate_horizon

COLUMNS=['symbol','date','horizon_days','entry_close','p_up','outcome_return','direction_correct']

def update_journal(path:str|Path,symbol:str,prices:pd.DataFrame,
                   probability:float|None,horizon_days:int=HORIZON):
    h=validate_horizon(horizon_days)
    path=Path(path)
    if path.exists():
        log=pd.read_csv(path,dtype={'symbol':str,'date':str})
        if not set(COLUMNS).issubset(log.columns):
            # Legacy 21-day journal conversion without losing completed records.
            legacy=['symbol','date','entry_close','p_up','outcome_21d','direction_correct']
            if set(legacy).issubset(log.columns):
                log=log.rename(columns={'outcome_21d':'outcome_return'})
                log['horizon_days']=21
            else:raise ValueError('예측 기록 파일 열 구성이 올바르지 않습니다.')
        log=log[COLUMNS].copy()
    else:log=pd.DataFrame(columns=COLUMNS)
    if len(prices)<h+1:return log
    positions={pd.Timestamp(d).strftime('%Y-%m-%d'):i for i,d in enumerate(prices.index)}
    eligible=log[(log.symbol==symbol)&(pd.to_numeric(log.horizon_days)==h)&(log.outcome_return.isna())]
    for i,row in eligible.iterrows():
        at=positions.get(str(row.date))
        if at is not None and at+h<len(prices):
            realized=float(prices.Close.iloc[at+h]/float(row.entry_close)-1)
            log.loc[i,'outcome_return']=realized
            log.loc[i,'direction_correct']=int((float(row.p_up)>=.5)==(realized>0))
    if probability is not None:
        date=prices.index[-1].strftime('%Y-%m-%d')
        exists=((log.symbol==symbol)&(log.date==date)&(pd.to_numeric(log.horizon_days)==h)).any()
        if not exists:
            log.loc[len(log)]=[symbol,date,h,float(prices.Close.iloc[-1]),float(probability),float('nan'),float('nan')]
    path.parent.mkdir(parents=True,exist_ok=True)
    log.to_csv(path,index=False)
    return log
