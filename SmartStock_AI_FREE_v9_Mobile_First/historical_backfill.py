"""Historical news backfill with date-based provenance and no fake observations.

Data source: GDELT DOC 2.0 article list. We ingest only article titles and
GDELT's first-seen timestamps, not full-text or today's Yahoo snapshots.

Historical availability varies by provider and query. Failed/incomplete
requests are never converted into 'zero news' observations.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import json
import re
import time
import requests
import numpy as np
from intelligence import news_headline_score
from point_in_time import FEATURE_NAMES, safe_symbol, _utc

ROOT=Path(__file__).resolve().parent
ENDPOINT='https://api.gdeltproject.org/api/v2/doc/doc'


def as_date(s):
    return s if isinstance(s,date) and not isinstance(s,datetime) else date.fromisoformat(str(s)[:10])


def chunks(start, end, days=7):
    """Yield inclusive date chunks."""
    day=as_date(start); end=as_date(end)
    if day>end:raise ValueError('시작일은 종료일 이전이어야 합니다')
    while day<=end:
        e=min(end,day+timedelta(days=days-1))
        yield day,e
        day=e+timedelta(days=1)


def parse_seen(raw):
    """GDELT seendate is a first-discovered UTC time, NOT a future prediction."""
    if not raw:return None
    try:
        s=str(raw).replace('T','').replace('Z','').replace('-','').replace(':','')
        dt=datetime.strptime(s[:14],'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        return dt
    except (ValueError,TypeError):return None


def search_chunk(company_name, start, end, session=None, maxrecords=250):
    """Returns (articles, is_complete).

    A full result page cannot prove completeness; do not infer 0 articles on
    missing/saturated days. All requests have timeouts and proper quoting.
    """
    if not 1<=maxrecords<=250:raise ValueError('maxrecords range 1..250')
    query='"'+str(company_name).replace('"','').strip()+'"'
    if len(query)<4 or len(query)>130:raise ValueError('회사 이름을 확인하세요')
    st=datetime.combine(as_date(start),datetime.min.time(),timezone.utc)
    en=datetime.combine(as_date(end)+timedelta(days=1),datetime.min.time(),timezone.utc)-timedelta(seconds=1)
    params={'query':query,'mode':'artlist','format':'json','maxrecords':maxrecords,
            'startdatetime':st.strftime('%Y%m%d%H%M%S'),
            'enddatetime':en.strftime('%Y%m%d%H%M%S')}
    sess=session or requests
    r=sess.get(ENDPOINT,params=params,timeout=30,
               headers={'User-Agent':'SmartStock-FreeResearch/7.0 (personal noncommercial use)'})
    r.raise_for_status()
    data=r.json()
    if not isinstance(data,dict) or not isinstance(data.get('articles'),list):
        # Some GDELT zero-result JSON responses return {"articles": []},
        # but other shapes or errors must not be treated as a zero-news day.
        raise ValueError('GDELT 응답에 articles 목록이 없거나 응답 형식이 변경되었습니다.')
    return data['articles'], len(data['articles'])<maxrecords


def valid_titles_by_day(articles,start,end):
    """Per-day dedup and strictly first-seen timestamp filtering."""
    st=as_date(start); en=as_date(end)
    by_day=defaultdict(dict)
    for item in articles:
        if not isinstance(item,dict):continue
        seen=parse_seen(item.get('seendate'))
        if seen is None or not st<=seen.date()<=en:continue
        title=str(item.get('title') or '').strip()
        if not title:continue
        url=str(item.get('url') or '')[:1000]
        unique=url or title.casefold()
        by_day[seen.date()][unique]={'title':title,'url':url,'first_seen':seen.isoformat()}
    return {day:list(rows.values()) for day,rows in by_day.items()}


def _append_historical(path, symbol, day, news, *, provider='GDELT_DOC2'):
    """Only provider-first-seen dated records; captured_at remains NOW.

    A historical record uses known_at_utc (day 23:59:59), distinct from
    captured_at_utc when it was downloaded. join_observations shifts it to the
    next trading day. The entire day is closed to avoid intra-day look-ahead.
    """
    score=news_headline_score([i['title'] for i in news]) if news else []
    nums=[float(s['score']) for s in score if 'score' in s]
    features={k:None for k in FEATURE_NAMES}
    features['news_sentiment']=float(np.mean(nums)) if nums else 0.0
    features['news_count']=float(len(news))
    known=datetime.combine(day,datetime.max.time(),timezone.utc).replace(microsecond=0)
    rec={'symbol':safe_symbol(symbol),'origin':'historical_verified',
         'captured_at_utc':datetime.now(timezone.utc).isoformat(),
         'known_at_utc':known.isoformat(),
         'source_provider':provider,
         'source_url':'https://www.gdeltproject.org/',
         'source_status':{'news':f'GDELT archived title metadata ({len(news)} matches); first-seen based, not full articles'},
         'features':features,
         'news_sources':[{'url':n['url'],'first_seen':n['first_seen']} for n in news[:12]]}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n')
    return rec


def existing_verified_days(path,symbol):
    days=set()
    if not Path(path).exists():return days
    for row in Path(path).read_text(encoding='utf-8').splitlines():
        try:
            r=json.loads(row)
            if r.get('symbol')==safe_symbol(symbol) and r.get('origin')=='historical_verified' and r.get('source_provider')=='GDELT_DOC2':
                days.add(_utc(r['known_at_utc']).date())
        except (json.JSONDecodeError,KeyError,TypeError,ValueError):pass
    return days


def backfill_news(symbol,company_name,start,end,*,path=None,session=None,
                  request_budget=25,chunk_days=7,throttle_seconds=0):
    """Backfill public historical headlines in bounded chunks.

    If a range is saturated, split until a complete daily query exists. Never
    treat failed or saturated queries as absence of news. Respect request budget.
    """
    symbol=safe_symbol(symbol)
    target=Path(path or ROOT/'data'/'context_snapshots.jsonl')
    current=existing_verified_days(target,symbol)
    tasks=list(chunks(start,end,chunk_days))[::-1]  # Latest first for recent relevance.
    # Prioritize the most recent dates but maintain chronologically honest known_at.
    saved=0;requests_made=0;errors=[];incomplete=[]
    while tasks and requests_made<request_budget:
        first,last=tasks.pop(0)
        if all(first+timedelta(days=i) in current for i in range((last-first).days+1)):
            continue
        try:
            articles,complete=search_chunk(company_name,first,last,session)
            requests_made+=1
            if not complete:
                if first<last:
                    mid=first+timedelta(days=(last-first).days//2)
                    # Split to ensure no silently lost news, process recent dates first.
                    tasks.insert(0,(first,mid))
                    tasks.insert(0,(mid+timedelta(days=1),last))
                else:
                    incomplete.append(str(first))
                continue
            by_day=valid_titles_by_day(articles,first,last)
            for day in (first+timedelta(days=i) for i in range((last-first).days+1)):
                if day in current:continue
                _append_historical(target,symbol,day,by_day.get(day,[]))
                current.add(day);saved+=1
        except (requests.RequestException,ValueError,KeyError) as ex:
            requests_made+=1
            errors.append(f'{first}~{last}: {type(ex).__name__} {str(ex)[:90]}')
        if throttle_seconds:time.sleep(throttle_seconds)
    return {'symbol':symbol,'company':company_name,'saved_days':saved,
            'requests_made':requests_made,'pending_ranges':len(tasks),
            'errors':errors,'saturated_days':incomplete,
            'coverage_days':len(current),'partial':bool(tasks or errors or incomplete)}


def main():
    ap=argparse.ArgumentParser(description='무료 과거 뉴스 검색 및 날짜 보존형 수집')
    ap.add_argument('--symbol',required=True)
    ap.add_argument('--company',required=True,help='정확한 회사 영문명 (예: CoreWeave)')
    ap.add_argument('--start',default='2024-01-01')
    ap.add_argument('--end',default=date.today().isoformat())
    ap.add_argument('--requests',type=int,default=25)
    ap.add_argument('--pause',type=float,default=1.0)
    args=ap.parse_args()
    result=backfill_news(args.symbol,args.company,args.start,args.end,
                         request_budget=max(1,min(args.requests,100)),
                         throttle_seconds=max(0,args.pause))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
