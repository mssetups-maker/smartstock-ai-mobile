"""Free best-effort snapshot acquisition; no real-time market data guarantees."""
from __future__ import annotations
import argparse
import os
from datetime import datetime,timezone
from pathlib import Path
from point_in_time import append_snapshot,normalized_features,safe_symbol
from intelligence import news_headline_score
from options_engine import analyze_options,pick_expiry

ROOT=Path(__file__).resolve().parent


def _headlines(ticker_obj):
    """Yahoo news can be unavailable; never fabricate headlines."""
    try:
        items=ticker_obj.news or []
        rows=[]
        for entry in items[:20]:
            item=entry.get('content',entry) if isinstance(entry,dict) else {}
            title=item.get('title') if isinstance(item,dict) else None
            if title:rows.append(str(title))
        return rows[:15]
    except Exception:return []


def collect_one(symbol:str, root:Path=ROOT, factory=None, now=None):
    symbol=safe_symbol(symbol)
    if factory is None:
        import yfinance as yf
        factory=yf.Ticker
    t=factory(symbol)
    observed={}; status={}; expiration=None
    try:
        opts=pick_expiry(list(t.options))
        if not opts:raise LookupError('옵션 상장·만기 데이터 없음')
        chains=t.option_chain(opts)
        hist=t.history(period='5d',interval='1d',auto_adjust=False,actions=False)
        if hist is None or hist.empty:raise ValueError('현재 기초자산 가격 없음')
        spot=float(hist['Close'].dropna().iloc[-1])
        rep=analyze_options(spot,chains.calls,chains.puts,opts)
        expiration=opts
        observed.update(normalized_features(option_report=rep))
        status['options']='조회 성공, 실시간 보장 없음'
    except Exception as ex:status['options']=f'조회 실패: {type(ex).__name__}: {str(ex)[:95]}'
    titles=_headlines(t)
    if titles:
        observed.update({k:v for k,v in normalized_features(news_titles=titles).items() if v is not None})
        status['news']=f'제목 {len(titles)}개 기록 (본문 없음)'
    else:status['news']='뉴스 제목 수집 불가'
    try:
        info=t.info
        if not isinstance(info,dict) or not info:raise ValueError('기초 재무정보 없음')
        observed.update({k:v for k,v in normalized_features(financial_info=info).items() if v is not None})
        status['finance']='최근 제공 정보 기록 (과거 재무 수정 이력 없음)'
    except Exception as ex:status['finance']=f'조회 실패: {type(ex).__name__}: {str(ex)[:95]}'
    if not any(v is not None for v in observed.values()):
        raise RuntimeError(f'{symbol}: 유효한 관측자료를 하나도 얻지 못했습니다. 기록하지 않았습니다. {status}')
    path=Path(root)/'data'/'context_snapshots.jsonl'
    record=append_snapshot(path,symbol,observed,status,captured_at=now or datetime.now(timezone.utc),expiry=expiration)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--symbols',nargs='*')
    args=parser.parse_args()
    symbols=args.symbols or (ROOT/'config'/'watchlist.txt').read_text(encoding='utf-8').splitlines()
    for sym in symbols[:15]:
        if not sym.strip() or sym.strip().startswith('#'):continue
        try:
            result=collect_one(sym)
            print('SNAPSHOT',result['symbol'],result['captured_at_utc'],result['source_status'])
        except Exception as e:print('UNAVAILABLE',sym,type(e).__name__,str(e)[:160])

if __name__=='__main__':main()
