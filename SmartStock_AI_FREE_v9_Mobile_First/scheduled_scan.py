"""Daily free research refresh for watched symbols and selectable horizons.

Price-based ML only; no trading. Labels are entered into persistent CSV only
when the relevant subsequent trading days have actually elapsed.
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from engine import clean_prices,indicators,analyze,validate_horizon
from journal import update_journal
from research import adaptive_signals
from strict_validation import audit_history
from risk_engine import data_audit
from alerts import send_alerts_if_enabled
from point_in_time import read_snapshots
from context_model import assess_context
ROOT=Path(__file__).resolve().parent


def run_scan(symbols:list[str],root:Path=ROOT,fetcher=None,horizons:list[int]|None=None):
    if fetcher is None:
        import yfinance as yf
        fetcher=lambda x: yf.Ticker(x).history(period='10y',interval='1d',auto_adjust=True,actions=False,timeout=18)
    horizons=[validate_horizon(int(h)) for h in (horizons if horizons is not None else [21])]
    if len(horizons)>10:raise ValueError('정기 분석 기간은 최대 10개를 지정하세요.')
    root.mkdir(parents=True,exist_ok=True)
    report=['# SmartStock AI FREE v5.0: 정기 재분석 및 과거검증',
            f'기준 UTC: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}',
            '과거 워크포워드 검증을 거친 조건부 신호만 표시. 과거 성적이 미래를 보장하지 않으며 실제 매매 서비스가 아닙니다.']
    results=[]
    for symbol in symbols:
        symbol=symbol.upper().strip()
        if not symbol or symbol.startswith('#'):continue
        try:
            raw=clean_prices(fetcher(symbol))
            if len(raw)<320:raise ValueError(f'자료가 부족합니다: {len(raw)}거래일')
            df=indicators(raw)
            quality=data_audit(raw)
            archived=read_snapshots(root/'data'/'context_snapshots.jsonl',symbol)
            for h in horizons:
                try:
                    res=analyze(df,horizon_days=h)
                    audit=audit_history(raw,h,res.probability,max_checks=22)
                    if quality.status=='차단':
                        audit.recommendation='판단 보류'
                        audit.explanation='오래되었거나 유효하지 않은 가격 데이터'
                    context=assess_context(raw,archived,horizon_days=h)
                    journal=update_journal(root/'data'/'predictions.csv',symbol,raw,res.probability,h)
                    past=journal[(journal.symbol==symbol)&(pd.to_numeric(journal.horizon_days)==h)&journal.direction_correct.notna()]
                    verified=f'{past.direction_correct.mean():.1%}' if len(past) else '결과 미확정'
                    out={'symbol':symbol,'horizon_days':h,
                         'price_date':raw.index[-1].strftime('%Y-%m-%d'),
                         'data_status':quality.status,
                         'close':round(float(raw.Close.iloc[-1]),4),
                         'raw_up_score':round(res.probability,4) if res.probability is not None else None,
                         'walk_forward_cases':audit.evaluations,
                         'past_buy_cases':audit.evaluated_long_signals,
                         'past_buy_hit_rate':audit.long_hit_rate,
                         'journal_settled':len(past),'journal_hit_rate':verified,
                         'conservative_decision':audit.recommendation,
                         'context_status':context.status,
                         'context_score':context.score,
                         'context_validation_cases':context.holdout_cases,
                         'option_history_bars':context.coverage['option_coverage'],
                         'news_history_bars':context.coverage['news_coverage']}
                    results.append(out)
                    report.append(f'### {symbol} ({h}거래일)\n\n'
                                  f'종가 {out["close"]:,.2f} / 보수적 판단: **{audit.recommendation}** / '
                                  f'과거 검증 {audit.evaluations}건 / 검증된 매수 신호 {audit.evaluated_long_signals}건 / '
                                  f'매수신호 적중률 {audit.long_hit_rate:.1%}' if audit.long_hit_rate is not None
                                  else f'### {symbol} ({h}거래일)\n\n**판단 보류:** 충분한 과거 매수 신호 없음')
                    report.append(f'보조 모델: **{context.status}** / 옵션 관측 {context.coverage["option_coverage"]}개, 뉴스 관측 {context.coverage["news_coverage"]}개')
                    print('OK',symbol,h,'OOS',audit.evaluations,'DECISION',audit.recommendation,'CONTEXT',context.status)
                except Exception as inner:
                    report.append(f'### {symbol} / {h}거래일: 분석 실패 ({str(inner)[:140]})')
                    print('FAIL',symbol,h,type(inner).__name__,str(inner)[:140])
        except Exception as e:
            report.append(f'### {symbol}: 데이터 조회 실패 ({str(e)[:140]})')
            print('FAIL',symbol,type(e).__name__,str(e)[:140])
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports'/'latest.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    pd.DataFrame(results).to_csv(root/'reports'/'latest.csv',index=False)
    # Opt-in Telegram alerts only when bot credentials are provided in Secrets.
    alert_state=send_alerts_if_enabled(results,root/'data')
    print('OPTIONAL_ALERTS:',alert_state['status'],'sent',alert_state['sent'])
    return results

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--symbols',nargs='*')
    a.add_argument('--horizons',nargs='*',type=int)
    args=a.parse_args()
    names=args.symbols or (ROOT/'config'/'watchlist.txt').read_text(encoding='utf-8').splitlines()
    config=ROOT/'config'/'horizons.txt'
    periods=args.horizons or ([int(x) for x in config.read_text().replace(',', '\n').splitlines() if x.strip().isdecimal()] if config.exists() else [5,21,63])
    run_scan(names,horizons=periods)
