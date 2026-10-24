import json
from datetime import date,datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from historical_backfill import backfill_news, parse_seen, search_chunk,valid_titles_by_day,existing_verified_days
from point_in_time import read_snapshots,join_observations
from sec_backfill import save_sec_filings, load_sec_filings


class FakeResponse:
    def __init__(self,content):self.content=content
    def raise_for_status(self):pass
    def json(self):return self.content

class FakeGdelt:
    def __init__(self):self.params=[]
    def get(self,url,params,timeout,headers):
        self.params.append(params)
        return FakeResponse({'articles':[
          {'title':'CoreWeave reports strong growth','url':'https://news.example/1','seendate':'20260915T151500Z'},
          {'title':'CoreWeave reports strong growth','url':'https://news.example/1','seendate':'20260915T151500Z'},
          {'title':'CoreWeave faces a loss','url':'https://news.example/2','seendate':'20260916T151500Z'}
        ]})


def test_first_seen_news_backfill_does_not_use_download_date(tmp_path):
    path=tmp_path/'snap.jsonl'
    fake=FakeGdelt()
    result=backfill_news('CRWV','CoreWeave','2026-09-15','2026-09-17',
                         path=path,session=fake,request_budget=2,chunk_days=3)
    assert result['saved_days']==3 and not result['partial']
    assert existing_verified_days(path,'CRWV')=={date(2026,9,15),date(2026,9,16),date(2026,9,17)}
    raws=[json.loads(x) for x in path.read_text().splitlines()]
    assert all(x['origin']=='historical_verified' for x in raws)
    assert raws[0]['captured_at_utc']!=raws[0]['known_at_utc']
    archive=read_snapshots(path,'CRWV')
    idx=pd.to_datetime(['2026-09-15','2026-09-16','2026-09-17','2026-09-18','2026-09-21'])
    joined=join_observations(idx,archive)
    assert np.isnan(joined.loc['2026-09-15','news_count'])
    assert joined.loc['2026-09-16','news_count']==1
    assert joined.loc['2026-09-17','news_count']==1
    assert joined.loc['2026-09-18','news_count']==0  # Complete zero-news day
    before=joined.copy()
    # Current snapshot must not rewrite historical first-seen dates.
    from point_in_time import append_snapshot
    append_snapshot(path,'CRWV',{'news_count':45}, {},captured_at='2026-10-23T15:00:00+00:00')
    after=join_observations(idx,read_snapshots(path,'CRWV'))
    pd.testing.assert_frame_equal(before,after)
    result2=backfill_news('CRWV','CoreWeave','2026-09-15','2026-09-17',
                          path=path,session=fake,request_budget=2,chunk_days=3)
    assert result2['saved_days']==0


def test_gdelt_api_failures_not_fake_zero_day(tmp_path):
    class Bad:
        def get(self,*a,**kw):return FakeResponse({'error':'rate limited'})
    path=tmp_path/'j.jsonl'
    r=backfill_news('NVDA','NVIDIA','2026-01-05','2026-01-11',path=path,session=Bad())
    assert r['partial'] and r['saved_days']==0 and r['errors']
    assert not path.exists()


def test_saturated_article_chunk_does_not_silently_count_zeros(tmp_path):
    class Saturated:
        def get(self,*a,**kw):
            d=a[1] if len(a)>1 else kw['params']
            day=d['startdatetime'][:8]
            return FakeResponse({'articles':[{'title':'Too many','seendate':day+'T120000Z','url':f'https://news.example/{i}'} for i in range(250)]})
    path=tmp_path/'j.jsonl'
    r=backfill_news('AAPL','Apple','2026-01-05','2026-01-05',path=path,session=Saturated(),request_budget=1)
    assert r['saved_days']==0 and r['saturated_days']


def test_news_historic_date_filter():
    rows=[{'title':'Past','seendate':'20200101T111100Z'},
          {'title':'Real','seendate':'20261022T081100Z'},
          {'title':'No time','seendate':'bad'}]
    ans=valid_titles_by_day(rows,'2026-10-22','2026-10-22')
    assert len(ans[date(2026,10,22)])==1
    assert parse_seen('garbage') is None


def test_sec_cache_real_filed_dates(tmp_path):
    class SECResponse:
        def __init__(self,data):self.data=data
        def json(self):return self.data
        def raise_for_status(self):return None
    class SECSession:
        def get(self,url,headers,timeout):
            if 'company_tickers' in url:
                return SECResponse({'0':{'ticker':'CRWV','cik_str':555}})
            return SECResponse({'facts':{'us-gaap':{'Assets':{'units':{'USD':[
                {'form':'10-Q','filed':'2026-08-10','end':'2026-06-30','val':100}
            ]}}}}})
    r=save_sec_filings('CRWV',root=tmp_path,user_agent='Researcher contact@example.org',session=SECSession())
    assert r['filings']==1
    saved=load_sec_filings('CRWV',tmp_path)
    assert str(saved.iloc[0]['filed'].date())=='2026-08-10'
