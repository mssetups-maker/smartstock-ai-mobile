from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_decision import decide, wave_candidates

NOW = datetime(2026,11,3,9,0,tzinfo=timezone(timedelta(hours=9)))
CONTEXT = {'symbol':'TSLA', 'timeframe':'1d','horizon_bars':5,
           'signal_bucket':'v2:high-volume-momentum','model_version':'v2.1',
           'asof':NOW.isoformat()}


def candidate(id, title, outcome=None, score=None, valid=True, data_valid=True):
    return {'id': id, 'title': title, 'outcome':outcome, 'rank_score':score,
            'valid': valid, 'data_valid':data_valid}


def records(up=60, down=25, sideways=15):
    samples=[]
    for j, name in enumerate(['상승']*up+['하락']*down+['횡보']*sideways):
        p=NOW-timedelta(days=(j+10)*8)
        row=dict(CONTEXT)
        row.update({'event_id':str(j), 'prediction_time':p.isoformat(),
                    'resolved_time':(p+timedelta(days=7)).isoformat(),
                    'actual_outcome':name,'split':'out_of_sample','non_overlapping':True})
        row.pop('asof')
        samples.append(row)
    return samples


def test_wrong_top_removed_and_valid_alternative_promoted():
    c=[candidate('wrong','잘못된 상승', '상승',99,valid=False),
       candidate('valid','새 분석 상승','상승',76)]
    out=decide(c,CONTEXT)
    assert out['top']['id']=='valid'
    assert len(out['rejected'])==1
    assert out['probabilities'] is None


def test_search_finds_replacement_after_initial_invalidation():
    out=decide([candidate('old','옛 파동',valid=False)],CONTEXT,
               additional_searches=[('다른 기점 재탐색',[candidate('new','새 파동',score=71)])])
    assert out['top']['id']=='new'
    assert out['searched']==['기본 분석','다른 기점 재탐색']


def test_no_valid_candidate_cannot_fabricate_answer():
    out=decide([candidate('wrong','틀린 답',score=99,valid=False)],CONTEXT,
               historical_cases=records())
    assert out['top'] is None and out['probabilities'] is None


def test_insufficient_evidence_never_displays_percent():
    out=decide([candidate('up','상승',outcome='상승',score=78)],CONTEXT,
               historical_cases=records(20,10,5))
    assert out['top']['id']=='up'
    assert out['probabilities'] is None
    assert out['sample_count']==35


def test_mature_independent_out_of_sample_cases_get_probabilities():
    out=decide([candidate('down','하락','하락',91),
                candidate('up','상승','상승',65),
                candidate('flat','횡보','횡보',70)],CONTEXT,
               historical_cases=records())
    assert out['top']['id']=='up'
    assert out['probabilities']=={'상승':.6,'하락':.25,'횡보':.15}
    assert sum(out['probabilities'].values())==1
    assert out['status']=='확률 검증 후보'


def test_future_and_in_sample_records_are_never_counted():
    good=records(20,20,20)
    leakage=records(50,0,0)
    for row in leakage:
        row['event_id']='leak-'+row['event_id']
        row['resolved_time']=(NOW+timedelta(days=3)).isoformat()
    for row in good[:20]: row['split']='in_sample'
    out=decide([candidate('up','상승','상승')],CONTEXT,good+leakage)
    assert out['sample_count']==40
    assert out['probabilities'] is None


def test_mismatch_market_regime_version_and_overlaps_rejected():
    r=records()
    for item in r[:30]:item['signal_bucket']='different-regime'
    for item in r[30:50]:item['model_version']='v1'
    for item in r[50:80]:item['non_overlapping']=False
    out=decide([candidate('a','상승','상승')],CONTEXT,r)
    assert out['sample_count']==20
    assert out['probabilities'] is None


def test_wave_candidate_never_becomes_forecast_probability():
    wave={'status':'후보','phase':'상승 충격파', 'order':5,
          'notes':'규칙 검증','nodes':[{'파동':'3'}]}
    out=decide(wave_candidates(wave),CONTEXT,historical_cases=records())
    assert out['top'] is not None
    assert out['probabilities'] is None
    assert out['top']['rank_score'] is None


def test_invalid_wave_no_label_candidate():
    assert wave_candidates({'status':'판정 보류','nodes':[]})==[]


def test_duplicate_candidates_never_double_ranked():
    out=decide([candidate('dup','일단',score=55),candidate('dup','중복',score=99)],CONTEXT)
    assert out['top']['title']=='일단'
    assert len(out['rejected'])==1
