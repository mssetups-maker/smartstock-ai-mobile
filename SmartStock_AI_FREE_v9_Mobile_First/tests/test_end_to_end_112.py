"""End-to-end module integration without external market connections or UI."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import clean_prices,indicators,analyze
from strict_validation import audit_history
from wave_auto_recount import estimate_with_recount
from quality_guard import verify_and_rebuild,latest_signal,scenario_candidates
from historical_evidence import fixed_rule_history,model_history_from_audit,evidence_context,RULE_VERSION
from analysis_decision import decide


def test_end_to_end_real_module_chain_from_csv():
    root=Path(__file__).resolve().parents[1]
    raw=clean_prices(pd.read_csv(root/'demo_synthetic.csv',parse_dates=['Date'],index_col='Date'))
    df=indicators(raw)
    quality=verify_and_rebuild(raw,df)
    assert quality['status']=='정상'
    score=analyze(df,horizon_days=5).probability
    strict=audit_history(raw,5,score,max_checks=24)
    assert len(strict.rows)>=15
    current=latest_signal(df,volume_reliable=quality['volume_reliable'])
    candidates=scenario_candidates(current,score)
    assert len(candidates)==3
    history=fixed_rule_history(raw,df,5,'DEMO')
    context=evidence_context('DEMO',5,'rule:'+current['bucket'],RULE_VERSION,raw.index[-1])
    report=decide(candidates,context,history,min_samples=80)
    assert report['top'] is not None
    assert report['probabilities'] is None or report['sample_count']>=80
    model_rows=model_history_from_audit(strict.rows,raw,'DEMO',5)
    assert len(model_rows)==len(strict.rows)
    wave=estimate_with_recount(df,requested_order=5)
    assert wave['status'] in ('후보','판정 보류')
    if wave['status']=='판정 보류':assert not wave.get('nodes')
