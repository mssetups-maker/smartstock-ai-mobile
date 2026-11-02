"""Unsupervised, research-only candle motif discovery with chronological holdouts.

No algorithm can prove a pattern is new to the world or predict returns reliably.
This module clusters normalized OHLCV shapes, selects a direction using *past*
training outcomes, and evaluates the locked clusters on later, purged data.
Overlapping forward targets are de-duplicated in reporting. Never trade from this.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from math import comb, sqrt
import json
import math
import hashlib

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

VERSION = '2026-10-25-discovery-2-global-time-split'
WINDOWS = (8, 16)
PERIODS = {'stocks':(5,21), 'crypto_daily':(5,21), 'crypto_5min':(12,)}
MIN_TRAIN=18
MIN_HOLDOUT=12


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df,pd.DataFrame) or not {'Open','High','Low','Close'}.issubset(df.columns):
        return pd.DataFrame()
    cols=['Open','High','Low','Close']+(['Volume'] if 'Volume' in df else [])
    d=df[cols].copy()
    for col in cols: d[col]=pd.to_numeric(d[col],errors='coerce')
    if 'Volume' not in d:d['Volume']=0.
    d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=['Open','High','Low','Close'])
    d=d[(d[['Open','High','Low','Close']]>0).all(axis=1)
        & (d.High>=d[['Open','Close','Low']].max(axis=1))
        & (d.Low<=d[['Open','Close','High']].min(axis=1))]
    if d.empty:return d
    d['Volume']=d['Volume'].clip(lower=0).fillna(0)
    try:
        d.index=pd.to_datetime(d.index,utc=True)
    except (ValueError, TypeError):return pd.DataFrame()
    return d[~d.index.duplicated(keep='last')].sort_index()


def _features(d: pd.DataFrame, t: int, window: int) -> np.ndarray | None:
    """Only known bars <=t; comparable candle geometry independent of price units."""
    if t<window+20:return None
    b=d.iloc[t-window+1:t+1]
    prev=d.Close.iloc[t-window:t]
    close=b.Close.to_numpy(dtype=float)
    op=b.Open.to_numpy(dtype=float)
    high=b.High.to_numpy(dtype=float)
    low=b.Low.to_numpy(dtype=float)
    past=np.log(d.Close.iloc[t-window-19:t+1].to_numpy(dtype=float))
    vol=float(np.std(np.diff(past)))
    scale=max(vol,0.001)
    r=np.clip(np.log(close/prev.to_numpy(dtype=float))/scale,-6,6)
    body=np.clip((close-op)/np.maximum(high-low,1e-12),-1,1)
    upper=np.clip((high-np.maximum(op,close))/np.maximum(high-low,1e-12),0,1)
    lower=np.clip((np.minimum(op,close)-low)/np.maximum(high-low,1e-12),0,1)
    volumes=b.Volume.to_numpy(dtype=float)
    hist_vol=float(np.median(d.Volume.iloc[max(0,t-window-19):t-window+1].to_numpy(dtype=float)))
    if hist_vol>0:
        v=np.clip(np.log1p(volumes/hist_vol)/2.0,0,4)
    else:
        v=np.zeros(window)
    # Five channels. Training never sees any future bar.
    return np.concatenate([r,body,upper,lower,v]).astype(float)


def _examples(prices:dict[str,pd.DataFrame], window:int, horizon:int, cap:int=2300):
    """One calendar-time cutoff shared by ALL symbols in a market.

    A per-symbol 68% split is unsafe: a newly listed stock's training
    labels could be newer than an older stock's holdout predictions. This
    global cutoff prevents that cross-symbol look-ahead. In addition, a
    training sample's TARGET date must precede the cutoff, not only its
    feature-observation date.
    """
    prepared={}; all_dates=[]
    for sym, raw in sorted(prices.items()):
        d=normalize_ohlcv(raw)
        if len(d)<max(180,window+horizon*4):continue
        if len(d)>cap:d=d.tail(cap)
        prepared[sym]=d
        all_dates.extend(d.index)
    if not all_dates:return None,{}
    calendar=sorted(set(all_dates))
    if len(calendar)<max(180,window+horizon*4):return None,{}
    cutoff=calendar[int(len(calendar)*.68)]
    xx=[];yy=[];istrain=[];symbols=[];times=[];positions=[];splitmeta={}
    for sym,d in prepared.items():
        stride=max(1,horizon//4)
        train_n=test_n=0
        for t in range(window+20,len(d)-horizon,stride):
            train=d.index[t+horizon]<cutoff  # label finished before holdout clock
            test=d.index[t]>=cutoff
            if not (train or test):continue  # purge boundary-crossing labels
            x=_features(d,t,window)
            if x is None or not np.isfinite(x).all():continue
            future=float(np.log(d.Close.iloc[t+horizon]/d.Close.iloc[t]))
            xx.append(x);yy.append(future);istrain.append(train)
            symbols.append(sym);times.append(d.index[t].isoformat());positions.append(t)
            train_n+=int(train);test_n+=int(test)
        splitmeta[sym]={'bars':len(d),'first':d.index[0].isoformat(),
                        'last':d.index[-1].isoformat(),
                        'split':cutoff.isoformat(),
                        'training_cases':train_n,'holdout_cases':test_n,
                        'global_calendar_split':True}
    if not xx:return None,splitmeta
    return {'X':np.vstack(xx),'y':np.asarray(yy),'train':np.asarray(istrain,dtype=bool),
            'symbols':np.array(symbols),'times':times,
            'positions':np.asarray(positions),'split_utc':cutoff.isoformat()},splitmeta


def _effective_indices(ids, examples, horizon,window):
    """Make outcome observations non-overlapping within a ticker."""
    last={};chosen=[]
    # Sort by ticker then chronological position rather than cluster arrival order.
    for i in sorted(ids,key=lambda j:(examples['symbols'][j],examples['positions'][j])):
        sym=examples['symbols'][i];pos=int(examples['positions'][i])
        if pos-last.get(sym,-10**9) < horizon+window:continue
        chosen.append(int(i));last[sym]=pos
    return chosen


def _binomial_tail(k,n,p):
    """Exact 1-sided P(X>=k), null fixed using training split only."""
    p=float(np.clip(p,.001,.999))
    return min(1.,sum(comb(n,j)*(p**j)*((1-p)**(n-j)) for j in range(k,n+1)))


def _bh(p_values):
    m=len(p_values)
    order=sorted(range(m),key=lambda k:p_values[k]);q=[1.]*m;best=1.
    for rank in range(m-1,-1,-1):
        i=order[rank];best=min(best,p_values[i]*m/(rank+1));q[i]=best
    return q


def _shape_description(center,window):
    r=np.asarray(center[:window]);tail=r[-max(2,window//3):]
    direction='상승 우위' if r.sum()>0.5 else '하락 우위' if r.sum()<-0.5 else '횡보·혼조'
    taildir='최근 구간 상승' if tail.sum()>0.5 else '최근 구간 하락' if tail.sum()<-0.5 else '최근 구간 횡보'
    return f'{window}봉, {direction}, {taildir} 형태의 이름 없는 캔들 군집'


def discover_market(prices:dict[str,pd.DataFrame],market:str,windows=WINDOWS,horizons=None,
                    min_train=MIN_TRAIN,min_holdout=MIN_HOLDOUT):
    """Research summary. No pattern is promoted to production trading signals."""
    horizons=tuple(horizons or PERIODS[market])
    experiments=[]; models=[];splits={}
    for h in horizons:
        for w in windows:
            data,meta=_examples(prices,w,h)
            splits.update(meta)
            if data is None:continue
            train=np.where(data['train'])[0];test=np.where(~data['train'])[0]
            if len(train)<150 or len(test)<40:continue
            scaler=StandardScaler().fit(data['X'][train])
            X=scaler.transform(data['X'])
            n_clusters=max(3,min(8,int(len(train)/80)))
            model=MiniBatchKMeans(n_clusters=n_clusters,random_state=47,n_init=3,batch_size=256,max_iter=80)
            model.fit(X[train]);labels=model.predict(X)
            # All validation comparisons are frozen before the test labels/outcomes.
            # Baseline is calculated strictly from the training period.
            base_y=data['y'][_effective_indices(train,data,h,w)]
            if len(base_y)<10:continue
            p_up=float(np.clip((base_y>0).mean(),.05,.95))
            candidate=[]
            for k in range(n_clusters):
                tr=_effective_indices([i for i in train if labels[i]==k],data,h,w)
                te=_effective_indices([i for i in test if labels[i]==k],data,h,w)
                if len(tr)<min_train:continue
                mean_train=float(data['y'][tr].mean());direction=1 if mean_train>0 else -1
                hits=sum(direction*data['y'][te]>0) if te else 0
                mean_test=(float(np.mean(direction*data['y'][te])) if te else None)
                baseline=p_up if direction==1 else 1-p_up
                p_value=_binomial_tail(hits,len(te),baseline) if len(te)>=min_holdout else 1.
                mean_center=np.asarray(model.cluster_centers_[k]);distances=np.linalg.norm(X[train[labels[train]==k]]-mean_center,axis=1)
                centroid_raw=scaler.inverse_transform([mean_center])[0]
                uid=f'{market}-{w}-{h}-{hashlib.sha1(np.around(centroid_raw,1).tobytes()).hexdigest()[:8]}'
                candidate.append({
                    'id':uid,'window':w,'horizon':h,'cluster':int(k),
                    'name':_shape_description(centroid_raw,w),'direction':'상승' if direction==1 else '하락',
                    'train_cases':len(tr),'holdout_cases':len(te),
                    'holdout_symbols':len(set(data['symbols'][te])) if te else 0,
                    'train_mean_return_pct':round(mean_train*100,3),
                    'holdout_hit_rate':round(hits/len(te),4) if te else None,
                    'holdout_signed_avg_return_pct':round(mean_test*100,3) if mean_test is not None else None,
                    'baseline_hit_rate_train':round(baseline,4),
                    'p_value':round(p_value,6),'q_value':None,
                    'status':'검증 보류',
                    'center':np.around(mean_center,5).tolist(),
                    'distance_limit':round(float(np.quantile(distances,.95)),4) if len(distances)>5 else None,
                })
            qvals=_bh([e['p_value'] for e in candidate])
            for r,q in zip(candidate,qvals):
                r['q_value']=round(q,6)
                enough=(r['holdout_cases']>=min_holdout and r['holdout_symbols']>=2)
                useful=(r['holdout_hit_rate'] is not None and r['holdout_hit_rate']>=r['baseline_hit_rate_train']+.08
                        and r['holdout_signed_avg_return_pct'] is not None and r['holdout_signed_avg_return_pct']>0)
                if enough and useful and q<=.10:
                    r['status']='시간순 검증 통과 연구 후보 (실전 미적용)'
                elif enough:
                    r['status']='시간순 검증 미통과'
                experiments.append({key:value for key,value in r.items() if key not in ('center','distance_limit')})
                # Save all locked clusters for later matching, but never claim a probability.
                models.append({**r,'market':market,'scaler_mean':np.around(scaler.mean_,5).tolist(),
                               'scaler_scale':np.around(scaler.scale_,5).tolist()})
    # Correct for every cluster / window / horizon candidate examined in
    # this market, not just one individual KMeans run.
    corrected=_bh([e['p_value'] for e in experiments])
    for row,q in zip(experiments,corrected):
        row['q_value']=round(q,6)
        enough=(row['holdout_cases']>=min_holdout and row['holdout_symbols']>=2)
        useful=(row['holdout_hit_rate'] is not None and
                row['holdout_hit_rate']>=row['baseline_hit_rate_train']+.08 and
                row['holdout_signed_avg_return_pct'] is not None and
                row['holdout_signed_avg_return_pct']>0)
        row['status']=('시간순 검증 통과 연구 후보 (실전 미적용)' if enough and useful and q<=.10
                       else '시간순 검증 미통과' if enough else '검증 보류')
    by_id={e['id']:(e['q_value'],e['status']) for e in experiments}
    for row in models:
        row['q_value'],row['status']=by_id[row['id']]
    return {'market':market,'symbols':sorted(splits),'coverage':splits,
            'experiments':experiments,'models':models,
            'validated_research_candidates':sum(e['status'].startswith('시간순 검증 통과') for e in experiments),
            'total_holdout_cases':sum(e['holdout_cases'] for e in experiments)}


def latest_match(prices:pd.DataFrame,report:dict,market='stocks',horizon=21):
    """Matches last-known bars to frozen prototypes, no future prices used."""
    d=normalize_ohlcv(prices)
    if d.empty:return []
    candidates=[]
    market_result=(report.get('markets') or {}).get(market) or {}
    for r in market_result.get('models',[]):
        if r['horizon']!=horizon:continue
        x=_features(d,len(d)-1,r['window'])
        if x is None:continue
        mean=np.asarray(r['scaler_mean']);scale=np.asarray(r['scaler_scale']);center=np.asarray(r['center'])
        if not(len(x)==len(mean)==len(scale)==len(center)):continue
        dist=float(np.linalg.norm((x-mean)/np.maximum(scale,1e-8)-center))
        if r.get('distance_limit') is not None and dist>r['distance_limit']:continue
        candidates.append({k:r.get(k) for k in ('id','name','window','horizon','direction','status',
            'train_cases','holdout_cases','holdout_symbols','holdout_hit_rate',
            'baseline_hit_rate_train','q_value') }|{'distance':round(dist,3),'date':d.index[-1].isoformat()})
    return sorted(candidates,key=lambda a:a['distance'])[:3]


def research_batch(markets:dict[str,dict[str,pd.DataFrame]],timestamp=None):
    results={};errors=[]
    for market,frames in markets.items():
        if market not in PERIODS:continue
        try:results[market]=discover_market(frames,market)
        except Exception as e:errors.append({'market':market,'error':str(e)[:200]})
    return {'timestamp_utc':timestamp or datetime.now(timezone.utc).isoformat(),
            'engine_version':VERSION,'research_only':True,
            'not_trading_signals':True,
            'explanation':'모든 종목에 동일한 시간 절단점으로 학습/검증 분할; 완성된 미래 수익률만 평가; 겹치는 검증 결과 간격 제거; 다중검정 보정. 새로운 패턴의 발견과 실전 예측 개선은 별개.',
            'markets':results,'errors':errors,
            'status':'ok' if results and not errors else 'partial' if results else 'failed'}


def save_research(path:Path,report:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        try:_restore_ids(report,json.loads(path.read_text(encoding='utf-8')))
        except (ValueError,KeyError,TypeError):pass
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    audit=path.with_name('novel_patterns_history.jsonl')
    summary={'date':report['timestamp_utc'][:10],
             'status':report.get('status','unknown'),'markets':{k:{'experiments':len(v['experiments']),
             'validated':v['validated_research_candidates'],'holdout':v['total_holdout_cases']} for k,v in report['markets'].items()}}
    # One record per day; reruns do not inflate observations.
    history=[]
    if audit.exists():
        for line in audit.read_text(encoding='utf-8').splitlines():
            try:history.append(json.loads(line))
            except ValueError:pass
    history=[v for v in history if v.get('date')!=summary['date']]
    history.append(summary)
    audit.write_text('\n'.join(json.dumps(v,ensure_ascii=False) for v in history[-366:])+'\n',encoding='utf-8')


def _restore_ids(report:dict, previous:dict):
    """Keep labels across reruns only if their OHLCV prototype stays similar.

    Center ordering from unsupervised KMeans is arbitrary; IDs cannot be used
    as durable identifiers without this correspondence step.
    """
    for market,group in report.get('markets',{}).items():
        old=(previous.get('markets') or {}).get(market,{}).get('models',[])
        seen=set();replacement={}
        for model in group.get('models',[]):
            center=(np.asarray(model['center'])*np.asarray(model['scaler_scale'])+
                    np.asarray(model['scaler_mean']))
            choices=[]
            for prev in old:
                if (prev.get('window'),prev.get('horizon'),prev.get('direction')) != (model['window'],model['horizon'],model['direction']):continue
                if prev['id'] in seen:continue
                target=(np.asarray(prev['center'])*np.asarray(prev['scaler_scale'])+
                        np.asarray(prev['scaler_mean']))
                if len(target)!=len(center):continue
                dist=float(np.sqrt(np.mean((center-target)**2)))
                choices.append((dist,prev['id']))
            if choices:
                dist,old_id=min(choices)
                if dist<=.25:
                    replacement[model['id']]=old_id;seen.add(old_id)
        for model in group.get('models',[]):model['id']=replacement.get(model['id'],model['id'])
        for entry in group.get('experiments',[]):entry['id']=replacement.get(entry['id'],entry['id'])


def update_forward_journal(path:Path, report:dict, market_prices:dict[str,dict[str,pd.DataFrame]]):
    """Record a prospective forecast from the *previously known* bar and
    score it only after H real subsequent bars have arrived on later runs.

    Crucially, a record opened today cannot be evaluated today. No revision
    to historical entries is considered new evidence, and repeats are deduped.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    events=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                v=json.loads(line)
                if isinstance(v,dict):events.append(v)
            except ValueError:pass
    datasets={}
    for market,group in market_prices.items():
        if market=='crypto_5min':continue  # hourly data evaluated by separate intraday engine
        for ticker,frame in group.items():
            clean=normalize_ohlcv(frame)
            if not clean.empty:datasets[(market,ticker)]=clean
    # Score past records only, BEFORE adding the new ones.
    for item in events:
        if item.get('status')!='pending':continue
        d=datasets.get((item.get('market'),item.get('symbol')))
        if d is None:continue
        anchor=pd.Timestamp(item['observed_asof'])
        positions=np.flatnonzero(d.index==anchor)
        if not len(positions):continue
        target=int(positions[-1])+int(item['horizon'])
        if target>=len(d):continue
        actual=float(d.Close.iloc[target]);initial=float(item['observed_close'])
        signed=(actual/initial-1)*(1 if item['direction']=='상승' else -1)
        item.update(status='resolved',target_asof=d.index[target].isoformat(),
                    actual_close=round(actual,6),signed_return_pct=round(signed*100,4),
                    direction_correct=bool(signed>0))
    known={(i.get('market'),i.get('symbol'),i.get('observed_asof'),i.get('horizon')) for i in events}
    for market,group in datasets.items():
        market_name,symbol=market
        for horizon in PERIODS[market_name]:
            match=latest_match(group,report,market_name,horizon)
            if not match:continue
            p=match[0];asof=group.index[-1].isoformat()
            key=(market_name,symbol,asof,horizon)
            if key in known:continue
            events.append({'market':market_name,'symbol':symbol,'observed_asof':asof,
                'recorded_utc':report.get('timestamp_utc'),'observed_close':round(float(group.Close.iloc[-1]),6),
                'horizon':horizon,'pattern_id':p['id'],'direction':p['direction'],
                'pattern_validation_status':p['status'],'status':'pending',
                'research_only':True})
            known.add(key)
    # Retain all pending and most recent completed observations up to safe cap.
    finished=[e for e in events if e.get('status')=='resolved'][-12000:]
    pending=[e for e in events if e.get('status')!='resolved'][-4000:]
    kept=sorted(finished+pending,key=lambda x:x.get('recorded_utc') or '')
    path.write_text('\n'.join(json.dumps(e,ensure_ascii=False,default=str) for e in kept)+'\n',encoding='utf-8')
    return {'pending':len(pending),'resolved':len(finished),
        'evaluated_correct':sum(i.get('direction_correct',False) for i in finished),
        'note':'예측 당시 관측 기록만 이후 가격이 확정되면 채점; 검증되지 않은 연구 후보'}
