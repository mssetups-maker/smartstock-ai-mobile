"""Optional Telegram notification, opt-in via secret env vars; no auto trading."""
from __future__ import annotations
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request


def alert_candidates(rows, previous=None):
    previous=previous or {}
    new_state=dict(previous)
    alerts=[]
    for row in rows:
        code=f'{row["symbol"]}:{row["horizon_days"]}'
        current=f'{row.get("conservative_decision","판단 보류")}|{row.get("data_status","?")}'
        old=previous.get(code)
        # Send only after the state changes, and only when there was a previous run.
        if old is not None and old!=current:
            alerts.append(f'[{row["symbol"]}] {row["horizon_days"]}거래일 · {current} · 기준일 {row.get("price_date","미확인")}')
        new_state[code]=current
    return alerts,new_state


def send_alerts_if_enabled(rows, data_dir):
    token=os.environ.get('SMARTSTOCK_TELEGRAM_BOT_TOKEN','').strip()
    chat=os.environ.get('SMARTSTOCK_TELEGRAM_CHAT_ID','').strip()
    # Do not write new state when unavailable. Never log credentials.
    if not token or not chat:return {'status':'비활성: 사용자 Telegram 설정 없음','sent':0}
    path=Path(data_dir)/'notification_state.json'
    try:old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    except (ValueError,OSError):old={}
    candidates,state=alert_candidates(rows,old)
    sent=0;failed=False
    for msg in candidates[:12]:
        data=urllib.parse.urlencode({'chat_id':chat,'text':'SmartStock AI 상태 변경\n'+msg}).encode()
        req=urllib.request.Request(f'https://api.telegram.org/bot{token}/sendMessage',data=data,method='POST')
        try:
            with urllib.request.urlopen(req,timeout=10) as r:
                r.read(512)
            sent+=1
        except Exception:
            failed=True
    if not failed:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'status':'일부 전송 실패' if failed else '상태 갱신 완료','sent':sent}
