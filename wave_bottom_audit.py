"""Append prospective corrective-wave low *research observations*, never orders.

Each observation is recorded at the LAST COMPLETED bar as seen by the job.
Only subsequent completed bars settle 5/21-bar forward prices; no backfilled
'historical' entries or retroactive signals. No win-rate or live fills claimed.
"""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from wave_bottoms import detect_bottoms


def _read(path:Path)->list[dict]:
    if not path.exists():return []
    results=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            item=json.loads(line)
            if isinstance(item,dict):results.append(item)
        except ValueError:continue
    return results


def _save(path:Path, obj)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,default=str,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def update_bottom_audit(histories:dict[str,pd.DataFrame],*,root:Path,now:datetime|None=None,order:int=5)->dict:
    now=now or datetime.now(timezone.utc)
    root=Path(root)
    filename=root/'reports'/'wave_bottom_forward_journal.jsonl'
    records=_read(filename)
    registered={(r.get('symbol'),r.get('bar_date'),r.get('kind')) for r in records}
    errors=[]; found={}; new=0; settled_now=0
    for symbol,prices in histories.items():
        try:
            x=prices.sort_index()
            # Input must already be closed candles; the caller uses daily_market_state.
            report=detect_bottoms(x,order=order)
            found[symbol]=report['candidates']
            date=str(x.index[-1])[:10]
            close=float(x.Close.iloc[-1])
            for c in report['candidates']:
                # '관찰 구간' gets logged to measure the accuracy of early warnings.
                key=(symbol,date,c['kind'])
                if key in registered:continue
                records.append({'symbol':symbol,'bar_date':date,'kind':c['kind'],
                    'stage':c['status'],'as_observed_close':round(close,6),
                    'zone_low':c['zone_low'],'zone_high':c['zone_high'],
                    'observed_low':c['observed_low'],
                    'created_at_utc':now.isoformat(),'outcomes':{},'research_only':True})
                registered.add(key);new+=1
        except (ValueError,IndexError,TypeError,KeyError) as exc:
            errors.append({'symbol':symbol,'error':str(exc)[:180]})
    for r in records:
        symbol=r.get('symbol')
        if symbol not in histories:continue
        x=histories[symbol].sort_index()
        date=str(r.get('bar_date',''))
        ix=[i for i,stamp in enumerate(x.index) if str(stamp)[:10]==date]
        if not ix:continue
        start=ix[-1]
        # Calculate forward horizons only after observing >= h *new* bars.
        for h in (5,21):
            key=str(h)
            if key in (r.get('outcomes') or {}) or start+h>=len(x):continue
            entry=float(r['as_observed_close'])
            if entry<=0:continue
            slice_=x.iloc[start+1:start+h+1]
            r.setdefault('outcomes',{})[key]={
                'evaluated_bar_date':str(x.index[start+h])[:10],
                'return_pct':round((float(slice_.Close.iloc[-1])/entry-1)*100,5),
                'maximum_intraperiod_drawdown_pct':round((float(slice_.Low.min())/entry-1)*100,5),
                'hypothetical_close_to_close':True,
            }
            settled_now+=1
    # Retry/idempotency: one observation per symbol+bar+pattern, no rewriting
    # original candidate fields. Keep prospective history for reproducibility.
    filename.parent.mkdir(parents=True,exist_ok=True)
    if records:
        tmp=filename.with_name(filename.name+'.tmp')
        tmp.write_text('\n'.join(json.dumps(z,ensure_ascii=False,default=str) for z in records)+'\n',encoding='utf-8')
        tmp.replace(filename)
    total=sum(len(v) for v in found.values())
    payload={'timestamp_utc':now.isoformat(),'status':'ok' if not errors else 'partial',
             'new_candidates':new,'current_candidates':total,'tracked_symbols':len(found),
             'total_logged':len(records),'newly_evaluated_horizons':settled_now,
             'evaluated_5':sum('5' in z.get('outcomes',{}) for z in records),
             'evaluated_21':sum('21' in z.get('outcomes',{}) for z in records),
             'errors':errors,'by_symbol':found,
             'note':'기간 종료 전 성적을 미리 계산하지 않습니다. 매수 주문 또는 체결 결과가 아닌 종가 대비 참고 수익률입니다.'}
    _save(root/'reports'/'wave_bottom_latest.json',payload)
    return payload
