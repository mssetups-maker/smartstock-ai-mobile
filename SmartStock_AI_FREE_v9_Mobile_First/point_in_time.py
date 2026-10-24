"""Append-only, timestamped market-context archive and point-in-time feature join.

IMPORTANT: every snapshot is assumed known only from the NEXT market bar after
its UTC collection date. It cannot be backdated using article publication dates
or Yahoo's current quote. Missing historical archives are never fabricated.
"""
from __future__ import annotations
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from intelligence import news_headline_score

FEATURE_NAMES = (
    'pcr_oi','pcr_volume','atm_iv','iv_move','iv_skew',
    'news_sentiment','news_count',
    'financial_revenue_growth','financial_profit_margin','financial_roe',
    'financial_pe','financial_debt_cash',
)
NUMERIC_FIN = {
    'financial_revenue_growth':'revenueGrowth',
    'financial_profit_margin':'profitMargins',
    'financial_roe':'returnOnEquity',
    'financial_pe':'trailingPE',
}


def safe_symbol(symbol: str) -> str:
    name=symbol.strip().upper()
    if not re.fullmatch(r'[A-Z0-9.\-^]{1,22}',name):
        raise ValueError('지원하지 않는 종목코드입니다.')
    return name


def numerical(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def normalized_features(option_report=None, option_raw=None, news_titles=None, financial_info=None):
    """Only actual observations; unsupported keys remain null, never inferred."""
    out={k:None for k in FEATURE_NAMES}
    if option_report is not None:
        options={
            'pcr_oi':'oi_pcr','pcr_volume':'volume_pcr',
            'atm_iv':'implied_vol','iv_move':'implied_move_pct','iv_skew':'skew_diff',
        }
        for dst,src in options.items():out[dst]=numerical(getattr(option_report,src,None))
    if isinstance(option_raw,dict):
        for dest,src in [('pcr_oi','oi_pcr'),('pcr_volume','volume_pcr'),('atm_iv','implied_vol'),
                         ('iv_move','implied_move_pct'),('iv_skew','skew_diff')]:
            if out[dest] is None:out[dest]=numerical(option_raw.get(src))
    titles=[str(x) for x in (news_titles or []) if x and str(x).strip()]
    if titles:
        sentiment=news_headline_score(titles)
        scores=[x['score'] for x in sentiment]
        out['news_sentiment']=float(np.mean(scores))
        out['news_count']=float(len(titles))
    if isinstance(financial_info,dict):
        for dst,key in NUMERIC_FIN.items():out[dst]=numerical(financial_info.get(key))
        debt=numerical(financial_info.get('totalDebt'))
        cash=numerical(financial_info.get('totalCash'))
        if debt is not None and cash is not None and cash>0:
            out['financial_debt_cash']=max(0.,debt)/cash
    return out


def _utc(when=None):
    value=when or datetime.now(timezone.utc)
    if isinstance(value,str):
        value=datetime.fromisoformat(value.replace('Z','+00:00'))
    if value.tzinfo is None:raise ValueError('수집시간은 UTC 오프셋이 있어야 합니다.')
    return value.astimezone(timezone.utc)


def append_snapshot(path: str|Path, symbol: str, features:dict, source_status:dict,
                    *, captured_at=None, expiry=None, origin='observed') -> dict:
    """Append one observation to local JSONL. Caller persists via GitHub commit."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    record={'symbol':safe_symbol(symbol),'captured_at_utc':_utc(captured_at).isoformat(),
            'origin':origin,'option_expiry':expiry,'source_status':source_status,
            'features':{k:numerical(features.get(k)) for k in FEATURE_NAMES}}
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    return record


def read_snapshots(path: str|Path, symbol: str) -> pd.DataFrame:
    path=Path(path);symbol=safe_symbol(symbol)
    if not path.exists():return pd.DataFrame(columns=['observed_day',*FEATURE_NAMES])
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rec=json.loads(line)
            if rec.get('symbol')!=symbol:continue
            origin=rec.get('origin')
            if origin=='observed':
                # Original v6 observations must only use UTC captured time.
                date=_utc(rec['captured_at_utc']).date()
                event_time=rec['captured_at_utc']
            elif (origin=='historical_verified'
                  and rec.get('source_provider')=='GDELT_DOC2'
                  and rec.get('source_url')=='https://www.gdeltproject.org/'):
                # Verified historical GDELT first-seen metadata, assigned to the
                # completed UTC day; NEVER substitute download time as event time.
                date=_utc(rec['known_at_utc']).date()
                event_time=rec['known_at_utc']
            else:continue
            values={k:numerical(rec.get('features',{}).get(k)) for k in FEATURE_NAMES}
            rows.append({'observed_day':pd.Timestamp(date),'captured_at':event_time,**values})
        except (ValueError,KeyError,TypeError,json.JSONDecodeError):continue
    if not rows:return pd.DataFrame(columns=['observed_day',*FEATURE_NAMES])
    # Multiple observations a day may have partial outages. Keep the latest
    # actually observed non-null field for each day; still available ONLY
    # from the following market day. Do not synthesize absent values.
    frame=pd.DataFrame(rows).sort_values('captured_at')
    frame=frame.groupby('observed_day',as_index=False).agg(
        {k:(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan)
         for k in FEATURE_NAMES})
    return frame.sort_values('observed_day').reset_index(drop=True)


def join_observations(price_index, archives: pd.DataFrame, max_age_days=7):
    """For bar date D, use only snapshots captured on date < D.

    Non-trading days are handled by backward asof; 7-calendar-day staleness cap.
    Data collected on a given calendar day is never usable for that day's OHLCV.
    """
    idx=pd.DatetimeIndex(price_index).tz_localize(None)
    out=pd.DataFrame(index=idx, columns=list(FEATURE_NAMES), dtype='float64')
    if archives is None or archives.empty:return out
    right=archives.copy()
    for col in FEATURE_NAMES:
        if col not in right.columns:right[col]=np.nan
    right['observed_day']=pd.to_datetime(right['observed_day']).dt.normalize()
    right=right.drop_duplicates('observed_day',keep='first').sort_values('observed_day')
    left=pd.DataFrame({'bar_date':idx.normalize(),'position':range(len(idx))}).sort_values('bar_date')
    merged=pd.merge_asof(left,right,left_on='bar_date',right_on='observed_day',
                         allow_exact_matches=False,direction='backward')
    valid=(merged['bar_date']-merged['observed_day']).dt.days.le(max_age_days)
    for col in FEATURE_NAMES:
        merged[col]=pd.to_numeric(merged[col],errors='coerce').where(valid)
    merged=merged.sort_values('position')
    out.loc[:,list(FEATURE_NAMES)]=merged[list(FEATURE_NAMES)].to_numpy(dtype='float64')
    return out


def archive_stats(frame):
    return {'price_bars':len(frame),
            'option_coverage':int(frame['pcr_oi'].notna().sum()),
            'news_coverage':int(frame['news_sentiment'].notna().sum()),
            'finance_coverage':int(frame['financial_revenue_growth'].notna().sum())}
