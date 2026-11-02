"""Unattended *scheduled* market observations and safeguarded research.

This is not a continuous server, market-data redistributor or trading bot.
GitHub Actions calls it; all results include collection time and errors.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from crypto_transfer import COINS, get_candles, transfer_audit
from crypto_intraday import test_one_hour
from pattern_discovery import research_batch, save_research, update_forward_journal
from engine import clean_prices
from paper_audit import update_paper_audit
from forecast_forward import update_forecast_audit

ROOT=Path(__file__).resolve().parent

def _dump(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def _read_watchlist(root:Path=ROOT):
    f=root/'config'/'watchlist.txt'
    return [x.strip().upper() for x in f.read_text().splitlines() if x.strip() and not x.startswith('#')][:12]

def hourly(*,root:Path=ROOT,coin_getter=get_candles,stock_getter=None,now=None):
    now=now or datetime.now(timezone.utc)
    results=[]; failures=[]; intraday={}
    for coin in COINS:
        try:
            file=root/'data'/'crypto_5min'/f'{coin}.csv.gz'
            meta=root/'data'/'crypto_5min'/f'{coin}.backfill.json'
            old=pd.DataFrame()
            if file.exists():
                old=clean_prices(pd.read_csv(file,index_col=0,parse_dates=True))
            # Fetch a long block once, and then recent bars only.
            requested=2600 if not meta.exists() else 180
            candles=coin_getter(coin,unit='minutes/5',count=requested,now=now)
            if candles.empty: raise ValueError('완성된 캔들이 없습니다.')
            merged=clean_prices(pd.concat([old,candles])) if len(old) else clean_prices(candles)
            report=test_one_hour(merged)
            intraday[coin]=report
            # Commit at most one 5-minute training snapshot per UTC hour.
            current_hour=now.isoformat()[:13]
            should_persist=not file.exists() or not (root/'data'/'crypto_5min'/f'{coin}.last_hour.txt').exists() or (root/'data'/'crypto_5min'/f'{coin}.last_hour.txt').read_text()!=current_hour
            if should_persist:
                file.parent.mkdir(parents=True,exist_ok=True)
                merged.to_csv(file,compression='gzip',float_format='%.10g')
                (file.parent/f'{coin}.last_hour.txt').write_text(current_hour,encoding='utf-8')
                if not meta.exists():
                    _dump(meta,{'first_backfill_utc':now.isoformat(),
                                'requested':requested,'received':len(candles),
                                'first_available':merged.index[0].isoformat()})
            current=candles.tail(28)
            results.append({'market':coin,'close':round(float(current.Close.iloc[-1]),5),
               'last_candle_utc':current.index[-1].isoformat(),
               'last_1h_return_pct':round(float((current.Close.iloc[-1]/current.Close.iloc[max(0,len(current)-13)]-1)*100),3),
               'sample_count':len(current),'saved_5min_rows':len(merged),
               'intraday_research_status':report['status']})
        except Exception as e:
            failures.append({'source':coin,'error':str(e)[:180]})
    if intraday:
        _dump(root/'reports'/'crypto_intraday_latest.json',
              {'timestamp_utc':now.isoformat(),'research_only':True,
               'crypto_5min_evaluations':intraday})
    if stock_getter is None:
        import yfinance as yf
        stock_getter=lambda t: yf.Ticker(t).history(period='5d',interval='1h',auto_adjust=True,timeout=16)
    equities=[]
    for symbol in _read_watchlist(root)[:6]:
        try:
            raw=clean_prices(stock_getter(symbol))
            if raw.empty: raise ValueError('가격 기록이 없습니다.')
            equities.append({'symbol':symbol,'last_observed_bar':raw.index[-1].isoformat(),
                'close':round(float(raw.Close.iloc[-1]),4),'source':'무료 Yahoo: 지연 또는 불완전 가능'})
        except Exception as e:
            failures.append({'source':symbol,'error':str(e)[:180]})
    payload={'timestamp_utc':now.isoformat(),'mode':'15분 예약 감시 (실시간 아님)',
        'cryptocurrency':results,'stocks':equities,'failures':failures,
        'status':'ok' if len(results)+len(equities)>0 else 'failed',
        'last_successful_observations':len(results)+len(equities)}
    # Save new snapshot only once per UTC hour, otherwise avoid 96 Git commits/day.
    path=root/'data'/'monitor_latest.json'
    prior=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if (not prior or prior.get('timestamp_utc','')[:13]!=now.isoformat()[:13]
            or prior.get('status')=='failed'):
        _dump(path,payload)
    print(json.dumps(payload,ensure_ascii=False))
    if payload['status']=='failed':raise RuntimeError('주식과 코인 데이터를 모두 받지 못했습니다. 로그를 확인하세요.')
    return payload


def _read_cached_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return clean_prices(pd.read_csv(path, index_col=0, parse_dates=True))
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'기존 코인 가격 기록 오류: {exc}') from exc


def _append_daily_audit(path:Path, payload:dict):
    """Keep one compact immutable-looking summary per UTC date and mode.

    GitHub can rerun the daily schedule; upsert by date so retried jobs do
    not count as additional historical observations. Only audit metadata
    is saved here; data files hold the OHLCV training input separately.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    events=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                item=json.loads(line)
                if isinstance(item,dict): events.append(item)
            except ValueError:
                continue
    audit={'date_utc':payload['timestamp_utc'][:10],
           'timestamp_utc':payload['timestamp_utc'],
           'symbols_checked':len(payload['results']),
           'crypto_coverage':payload.get('crypto_coverage',{}),
           'qualified_candidates':sum(r.get('status','').startswith('연구상 성능 개선 후보') for r in payload['results']),
           'testable_experiments':sum('stock_holdout_brier' in r for r in payload['results']),
           'errors':payload.get('errors',[]),
           'results':[{k:v for k,v in r.items() if k in (
                'symbol','horizon_days','status','reason','stock_training_cases',
                'crypto_training_cases','stock_holdout_brier','stock_crypto_holdout_brier',
                'effective_holdout_cases')} for r in payload['results']]}
    events=[e for e in events if e.get('date_utc')!=audit['date_utc']]
    events.append(audit)
    # Bound file growth; one year's worth is adequate for first stage.
    path.write_text('\n'.join(json.dumps(e,ensure_ascii=False,default=str) for e in events[-366:])+'\n',encoding='utf-8')


def daily(*,root:Path=ROOT,coin_getter=get_candles,stock_getter=None,now=None):
    now=now or datetime.now(timezone.utc)
    reports=[];failure=[];crypto={};coverage={};stock_history={}
    for c in COINS:
        try:
            file=root/'data'/'crypto_daily'/f'{c}.csv'
            meta=root/'data'/'crypto_daily'/f'{c}.backfill.json'
            old=_read_cached_candles(file)
            # The previous 1300-day window began too late for long-history
            # stock research cutoffs. Backfill 3,800 bars ONCE, then fetch
            # only the recent bars in later runs. Exchange history may be
            # shorter than 3,800 because the pair was listed later.
            deep=not meta.exists()
            requested=3800 if deep else 10
            new=coin_getter(c,unit='days',count=requested,now=now)
            new=clean_prices(new)
            merged=clean_prices(pd.concat([old,new]).sort_index()) if len(old) else new
            if len(merged)<250:
                raise ValueError(f'일봉 자료 {len(merged)}개로 부족합니다.')
            crypto[c]=merged
            file.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(file,float_format='%.10g')
            if deep:
                _dump(meta,{'first_backfill_utc':now.isoformat(),
                    'requested':requested,'received':len(new),
                    'first_available':merged.index[0].isoformat(),
                    'last_available':merged.index[-1].isoformat()})
            coverage[c]={'rows':len(merged),'first':merged.index[0].isoformat(),
                'last':merged.index[-1].isoformat(),'deep_backfill':deep}
        except Exception as exc:
            failure.append({'source':c,'error':str(exc)[:180]})
    if stock_getter is None:
        import yfinance as yf
        stock_getter=lambda t: yf.Ticker(t).history(period='10y',interval='1d',auto_adjust=True,timeout=22)
    # Model comparisons remain in their own development + untouched holdout
    # windows. Results are research only; do NOT deploy them to live UI.
    horizons=(5,21,63)
    for symbol in _read_watchlist(root)[:12]:
        try:
            raw=clean_prices(stock_getter(symbol))
            if len(raw)<320:raise ValueError(f'주식 가격 자료가 {len(raw)}일뿐입니다.')
            stock_history[symbol]=raw
            for h in horizons:
                result=transfer_audit(raw,crypto,horizon=h)
                result['symbol']=symbol
                reports.append(result)
        except Exception as exc:
            failure.append({'source':symbol,'error':str(exc)[:180]})
    # Independent pattern-discovery research: shapes discovered without
    # future prices; candidate outcomes judged only in later, purged splits.
    # Research cannot modify or activate the live trading decision engine.
    novel_summary={'status':'not_run','errors':[]}
    try:
        experiment_markets={'stocks':stock_history,'crypto_daily':crypto}
        intraday_frames={}
        for c in COINS:
            f=root/'data'/'crypto_5min'/f'{c}.csv.gz'
            if f.exists():
                try:intraday_frames[c]=_read_cached_candles(f)
                except (OSError,ValueError,KeyError) as exc:
                    failure.append({'source':f'5min {c}','error':str(exc)[:180]})
        if intraday_frames:experiment_markets['crypto_5min']=intraday_frames
        new_research=research_batch(experiment_markets,timestamp=now.isoformat())
        save_research(root/'reports'/'novel_patterns_latest.json',new_research)
        prospective=update_forward_journal(root/'reports'/'novel_pattern_forward_journal.jsonl',new_research,experiment_markets)
        novel_summary={'status':new_research['status'],
            'markets':{k:{'clusters':len(v['experiments']),
                'validated_candidates':v['validated_research_candidates'],
                'holdout_cases':v['total_holdout_cases']} for k,v in new_research['markets'].items()},
            'errors':new_research['errors'],'forward_outcomes':prospective}
        if new_research['status'] in ('failed','partial'):
            failure.extend({'source':f"pattern_discovery/{r.get('market','?')}",
                            'error':str(r.get('error','연구 처리 불가'))[:180]}
                           for r in new_research.get('errors',[]))
            if not new_research.get('errors'):
                failure.append({'source':'pattern_discovery','error':'패턴 연구 결과가 완전하지 않습니다.'})
    except Exception as exc:
        novel_summary={'status':'failed','errors':[str(exc)[:180]]}
        failure.append({'source':'pattern_discovery','error':str(exc)[:180]})
    # Prospective 10.8 PAPER journal: decisions made from completed daily
    # candles are saved today; only future observed candles can settle them.
    # This is separate from research and cannot place orders or deploy a model.
    paper_summary={'status':'not_run'}
    try:
        paper_summary=update_paper_audit(stock_history,root=root,now=now)
        paper_summary['status']='partial' if paper_summary.get('errors') else 'ok'
        if paper_summary.get('errors'):
            failure.extend({'source':f"paper_audit/{item.get('symbol','?')}",
                            'error':str(item.get('error',''))[:180]}
                           for item in paper_summary['errors'])
    except Exception as exc:
        paper_summary={'status':'failed','error':str(exc)[:180]}
        failure.append({'source':'paper_audit','error':str(exc)[:180]})
    # A separate prospective ML SCORE journal. The recorded prediction
    # cannot be recomputed with a future-trained model; only later realized
    # prices settle its outcome. Never deploy a model on this basis alone.
    ml_summary={'status':'not_run'}
    try:
        ml_summary=update_forecast_audit(stock_history,root=root,now=now)
        ml_summary['status']='partial' if ml_summary.get('errors') else 'ok'
        if ml_summary.get('errors'):
            failure.extend({'source':f"forecast_forward/{item.get('symbol','?')}",
                            'error':str(item.get('error',''))[:180]}
                           for item in ml_summary['errors'])
    except Exception as exc:
        ml_summary={'status':'failed','error':str(exc)[:180]}
        failure.append({'source':'forecast_forward','error':str(exc)[:180]})
    # Prioritize the 21-session forecast in the existing phone status view.
    reports.sort(key=lambda r:(r.get('horizon_days')!=21, r.get('symbol',''),r.get('horizon_days',0)))
    payload={'timestamp_utc':now.isoformat(),
             'mode':'일일 주식·코인 전이학습: 5/21/63거래일, 독립 검증',
             'status':'ok' if reports and not failure else 'partial' if reports else 'failed',
             'integrated_into_live_stock_predictions':False,
             'why':'연구 결과만 저장하며, 독립 검증과 실제 미래 성과가 확인되기 전에는 실거래 모델을 교체하지 않습니다.',
             'crypto_coverage':coverage,'novel_pattern_research':novel_summary,
             'paper_strategy_audit':paper_summary,
             'forward_ml_audit':ml_summary,'horizons':list(horizons),
             'results':reports,'errors':failure}
    # A compact, independently checkable health record for the mobile UI.
    # Keep 'partial' distinct from 'ok': a green workflow step does not prove
    # every data source, optional research module, or model succeeded.
    health_record={
       'version':'10.9','timestamp_utc':now.isoformat(),
       'status':payload['status'],
       'stock_crypto_experiments':len(reports),
       'paper_strategy_status':paper_summary.get('status','not_run'),
       'ml_forward_status':ml_summary.get('status','not_run'),
       'pattern_research_status':novel_summary.get('status','not_run'),
       'failed_subsystems':failure[:40],
       'failure_count':len(failure),
       'model_auto_deployed':False,
       'note':'프로세스 완료는 예측 성능 향상이나 검증된 매수 신호를 뜻하지 않습니다.'}
    _dump(root/'reports'/'system_health_latest.json',health_record)
    _dump(root/'reports'/'crypto_transfer_latest.json',payload)
    _append_daily_audit(root/'reports'/'crypto_transfer_history.jsonl',payload)
    # Visible status summary separate from the enormous transfer-audit JSON.
    print('SMARTSTOCK_10_9_HEALTH '+json.dumps({
        'overall':payload['status'],
        'transfer_experiments':len(reports),
        'novel_patterns':novel_summary.get('status'),
        'paper_trading':paper_summary.get('status'),
        'forward_ml':ml_summary.get('status'),
        'data_issues':len(failure),
        'models_auto_deployed':False},ensure_ascii=False))
    if failure:
        print('::warning title=SmartStock 10.9 partial analysis::'+
              str(len(failure))+' data/model subsystem errors. Review report JSON; job is not a complete success.')
    print(json.dumps(payload,ensure_ascii=False,default=str))
    if not reports:
        raise RuntimeError('전이학습 실험을 실행하지 못했습니다. 데이터 연결을 확인하세요.')
    if ml_summary.get('status')=='failed':
        raise RuntimeError('10.8 AI 예측 사후평가 엔진이 실패했습니다. 기존 기록을 보존했습니다.')
    if paper_summary.get('status')=='failed':
        raise RuntimeError('10.8 모의매매 평가 엔진이 실패했습니다. 이전 거래 기록을 안전하게 보존했습니다.')
    if novel_summary.get('status')=='failed':
        raise RuntimeError('패턴 연구 엔진이 실패했습니다. reports/crypto_transfer_latest.json의 errors와 GitHub 실행 로그를 확인하세요.')
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['hourly','daily'],required=True)
    mode=p.parse_args().mode
    (hourly if mode=='hourly' else daily)()

if __name__=='__main__':main()
