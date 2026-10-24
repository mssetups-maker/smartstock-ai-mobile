from datetime import date
import numpy as np
import pandas as pd
from timeframe import TIMEFRAMES,timeframe_config,fetch_timeframe,chart
from v8_features import (reliability_calibration,classify_market,model_cost_stress,
   option_snapshot_changes,earnings_event_info,indicator_candidate_lab,watchlist_relative)
from alerts import alert_candidates


def prices(n=1100):
    r=np.random.default_rng(17)
    c=90*np.exp(np.cumsum(r.normal(.0002,.011,size=n)))
    idx=pd.bdate_range('2020-01-01',periods=n)
    df=pd.DataFrame({'Open':c*.999,'High':c*1.013,'Low':c*.987,'Close':c,
                     'Volume':r.integers(2000,40000,size=n)},index=idx)
    return df


def test_timeframe_mappings():
    assert TIMEFRAMES['1분봉'].interval=='1m'
    assert TIMEFRAMES['1시간봉'].interval=='1h'
    assert TIMEFRAMES['일봉'].interval=='1d'
    assert TIMEFRAMES['주봉'].interval=='1wk'
    try:timeframe_config('2초봉');assert False
    except ValueError:pass


def test_fetch_timeframe_mock():
    class T:
        def history(self,**kwargs):
            assert kwargs['interval']=='5m' and kwargs['period']=='1mo'
            return prices(60)
    assert len(fetch_timeframe(T(),'5분봉'))==60


def test_chart_with_stochastic_and_volume():
    fig=chart(prices(350),'1시간봉')
    assert sum(t.type=='candlestick' for t in fig.data)==1
    assert any('MA20' in (t.name or '') for t in fig.data)
    assert any((t.name or '')=='20K' for t in fig.data)


def test_calibration_not_force_probability():
    assert reliability_calibration([], .9)['calibrated_latest'] is None
    x=[.2,.4,.6,.8]*10
    ys=[0,0,1,1]*10
    rows=[{'날짜':f'2023-01-{i+1:02d}' if i<28 else f'2023-02-{i-27:02d}',
           '모델상승점수':xx,'실제상승':yy} for i,(xx,yy) in enumerate(zip(x,ys))]
    result=reliability_calibration(rows,.7)
    assert result['train']>=12 and result['test']>=10
    assert 0<=result['cal_brier']<=1


def test_no_calibration_when_y_all_same():
    rows=[{'날짜':f'2023-01-{i+1:02d}' if i<28 else f'2023-02-{i-27:02d}',
           '모델상승점수':.7,'실제상승':1} for i in range(30)]
    assert reliability_calibration(rows,.7)['calibrated_latest'] is None


def test_market_classification_uses_current_history_only():
    p=prices(300);x=classify_market(p)
    assert x['regime']!='자료 부족'
    q=p.copy();q.loc[p.index[-1],'Close']=float(q.Close.iloc[-1]*.15)
    assert classify_market(q)['regime']!=x['regime'] or classify_market(q)['high_vol']


def test_trade_cost_stress():
    rows=[{'신호':'매수 검토','실제수익률':.01},{'신호':'매수 검토','실제수익률':-.02}]
    x=model_cost_stress(rows,.3)
    assert x['cases']==2 and abs(x['avg_net']-(-.008))<.00001


def test_option_change_requires_two_actual_snapshots():
    f=pd.DataFrame({'observed_day':pd.to_datetime(['2026-10-01','2026-10-02']),
                    'pcr_oi':[.7,.9],'atm_iv':[.35,.41]})
    rows=option_snapshot_changes(f)
    assert any(abs(r['변화']-.2)<1e-8 for r in rows if r['항목']=='풋/콜 OI 비율')


def test_earnings_unknown_and_known():
    assert not earnings_event_info({},date(2026,10,23))['known']
    out=earnings_event_info({'Earnings Date':[pd.Timestamp('2026-10-26')]},date(2026,10,23))
    assert out['high_risk'] and out['days_to_event']==3


def test_candidate_lab_independent_period():
    out=indicator_candidate_lab(prices(1300),horizon=21)
    assert isinstance(out['rows'],list)
    for item in out['rows']:
        assert item['미사용 기간 신호 건수']>=5


def test_watchlist_comparison():
    out=watchlist_relative({'AAPL':prices(140),'NVDA':prices(190)})
    assert len(out)==2


def test_alerts_only_after_change():
    row={'symbol':'NVDA','horizon_days':21,'conservative_decision':'판단 보류','data_status':'정상','price_date':'2026-10-22'}
    assert alert_candidates([row])[0]==[]
    items,state=alert_candidates([row],{'NVDA:21':'조건부 매수 검토|정상'})
    assert len(items)==1 and '판단 보류' in items[0]


def test_interval_audit_is_never_certified():
    from v8_features import interval_coverage_audit
    result=interval_coverage_audit(prices(1500),horizon=21)
    assert 0<=result['historical_coverage']<=1
    assert result['cases']>=12
    assert result['upper_price']>=result['lower_price']


def test_error_pattern_descriptive():
    from v8_features import forecast_error_patterns
    rows=[{'모델상승점수':.8 if i%2 else .2,
           '실제상승':int(i%3==0),'상승추세':i%2==0,
           '시장국면':'고변동' if i%2 else '저변동'} for i in range(22)]
    result=forecast_error_patterns(rows)
    assert result['cases']==22
    assert len(result['groups'])>=3
