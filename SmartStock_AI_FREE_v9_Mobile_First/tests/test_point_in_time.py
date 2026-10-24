from datetime import datetime,timezone,timedelta
import json
import numpy as np
import pandas as pd
from point_in_time import normalized_features,append_snapshot,read_snapshots,join_observations
from sec_history import extract_filing_series,financial_features_asof


def test_only_next_trading_day(tmp_path):
    path=tmp_path/'facts.jsonl'
    append_snapshot(path,'NVDA',{'pcr_oi':2.5,'news_sentiment':1.0},{'news':'ok'},
                    captured_at=datetime(2026,10,23,23,tzinfo=timezone.utc))
    obs=read_snapshots(path,'NVDA')
    idx=pd.to_datetime(['2026-10-22','2026-10-23','2026-10-26','2026-10-29','2026-11-04'])
    data=join_observations(idx,obs,max_age_days=7)
    assert np.isnan(data.iloc[0].pcr_oi)
    assert np.isnan(data.iloc[1].pcr_oi)
    assert data.iloc[2].pcr_oi==2.5
    assert data.iloc[3].pcr_oi==2.5
    assert np.isnan(data.iloc[4].pcr_oi)


def test_future_snapshot_cannot_change_past(tmp_path):
    path=tmp_path/'x.jsonl'
    append_snapshot(path,'CRWV',{'pcr_oi':1},{},captured_at='2026-01-05T22:00:00+00:00')
    idx=pd.bdate_range('2026-01-01','2026-01-16')
    before=join_observations(idx,read_snapshots(path,'CRWV'))
    append_snapshot(path,'CRWV',{'pcr_oi':999},{},captured_at='2026-01-15T20:00:00+00:00')
    after=join_observations(idx,read_snapshots(path,'CRWV'))
    pd.testing.assert_frame_equal(before.loc[:'2026-01-15'],after.loc[:'2026-01-15'])


def test_unknown_is_not_zero(tmp_path):
    x=normalized_features(news_titles=['weak losses','beat growth'])
    assert x['pcr_oi'] is None
    assert x['news_count']==2
    path=tmp_path/'x.jsonl'
    append_snapshot(path,'MSFT',x,{},captured_at='2026-01-09T23:00:00+00:00')
    assert read_snapshots(path,'AAPL').empty


def test_filing_only_after_report():
    facts={'facts':{'us-gaap':{'RevenueFromContractWithCustomerExcludingAssessedTax':{'units':{'USD':[
        {'form':'10-K','filed':'2024-03-01','start':'2023-01-01','end':'2023-12-31','val':100},
        {'form':'10-K','filed':'2025-03-01','start':'2024-01-01','end':'2024-12-31','val':120},
        {'form':'10-K','filed':'2026-03-01','start':'2025-01-01','end':'2025-12-31','val':150}
    ]}}}}}
    frame=extract_filing_series(facts)
    dates=pd.to_datetime(['2025-02-25','2025-03-01','2025-03-03','2026-02-25'])
    y=financial_features_asof(dates,frame)
    assert np.isnan(y.loc['2025-02-25','sec_revenue_yoy'])
    assert np.isnan(y.loc['2025-03-01','sec_revenue_yoy'])
    assert abs(y.loc['2025-03-03','sec_revenue_yoy']-.2)<1e-12
    assert abs(y.loc['2026-02-25','sec_revenue_yoy']-.2)<1e-12


def test_context_model_refuses_unavailable_history():
    from context_model import assess_context
    path=str(__import__('pathlib').Path(__file__).resolve().parents[1]/'demo_synthetic.csv')
    prices=pd.read_csv(path,parse_dates=['Date'],index_col='Date').iloc[-500:]
    empty=pd.DataFrame()
    r=assess_context(prices,empty,horizon_days=21)
    assert r.score is None
    assert '기록 부족' in r.status


def test_context_model_runs_only_with_historic_observations(tmp_path):
    from context_model import assess_context
    path=str(__import__('pathlib').Path(__file__).resolve().parents[1]/'demo_synthetic.csv')
    prices=pd.read_csv(path,parse_dates=['Date'],index_col='Date').iloc[-900:]
    df=pd.DataFrame({
        'observed_day':pd.DatetimeIndex(prices.index[:-2]).normalize(),
        'pcr_oi':np.linspace(.8,1.2,len(prices)-2),
        'atm_iv':np.full(len(prices)-2,.4),
        'news_sentiment':np.linspace(-1,1,len(prices)-2),
    })
    result=assess_context(prices,df,horizon_days=5)
    assert result.holdout_cases>=8
    assert result.brier_context is not None
    assert result.brier_price_only is not None
    assert result.included_features[:3]==('pcr_oi','atm_iv','news_sentiment')


def test_multiple_same_day_captures_keep_real_observations(tmp_path):
    file=tmp_path/'facts.jsonl'
    append_snapshot(file,'NVDA',{'pcr_oi':1.4}, {},captured_at='2026-06-01T15:00:00+00:00')
    append_snapshot(file,'NVDA',{'news_sentiment':-2}, {},captured_at='2026-06-01T20:00:00+00:00')
    seen=read_snapshots(file,'NVDA')
    assert len(seen)==1
    assert seen.loc[0,'pcr_oi']==1.4
    assert seen.loc[0,'news_sentiment']==-2
    dates=pd.to_datetime(['2026-06-01','2026-06-02'])
    joined=join_observations(dates,seen)
    assert np.isnan(joined.iloc[0].pcr_oi)
    assert joined.iloc[1].news_sentiment==-2


def test_free_snapshot_collector_with_mock_provider(tmp_path):
    from archive_collector import collect_one
    from datetime import date,timedelta
    from types import SimpleNamespace
    import pandas as pd
    class Stub:
        @property
        def options(self):return [(date.today()+timedelta(days=30)).isoformat()]
        def option_chain(self,expiry):
            cols={'strike':[95,100,105],'openInterest':[60,100,45],
                  'volume':[8,10,6],'impliedVolatility':[.4,.4,.4],
                  'bid':[2,1,1],'ask':[3,2,2],'lastPrice':[2.5,1.5,1.5]}
            return SimpleNamespace(calls=pd.DataFrame(cols),puts=pd.DataFrame(cols))
        def history(self,**kw):return pd.DataFrame({'Close':[100.0]},index=pd.to_datetime(['2026-01-01']))
        @property
        def news(self):return [{'content':{'title':'Strong growth and record profit'}}]
        @property
        def info(self):return {'revenueGrowth':.34,'profitMargins':.12}
    fact=collect_one('NVDA',root=tmp_path,factory=lambda _sym:Stub(),now='2026-10-23T21:12:00+00:00')
    assert fact['features']['pcr_oi']==1
    assert fact['features']['financial_revenue_growth']==.34
    assert fact['features']['news_sentiment']>0
    from point_in_time import read_snapshots
    archive=read_snapshots(tmp_path/'data'/'context_snapshots.jsonl','NVDA')
    assert len(archive)==1
