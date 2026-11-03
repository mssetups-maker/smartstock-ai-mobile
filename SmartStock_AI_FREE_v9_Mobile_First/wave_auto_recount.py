"""Retry different confirmed-pivot scales before withholding Elliott labels.

This extends the unchanged 11.0 engine. It does not claim the alternative is
uniquely true, and never reuses an invalidated count as a valid replacement.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from elliott_waves import estimate_wave

SCALES=(3,5,8,12)


def estimate_with_recount(prices: pd.DataFrame, requested_order: int=5, lookback: int=420) -> dict[str,Any]:
    if requested_order not in SCALES:
        raise ValueError('허용 민감도: 3, 5, 8, 12')
    priority=[requested_order]+sorted((v for v in SCALES if v!=requested_order),key=lambda v:(abs(v-requested_order),v))
    discarded=[]
    first=None
    for level in priority:
        hypothesis=estimate_wave(prices,order=level,lookback=lookback)
        if first is None:first=hypothesis
        if hypothesis.get('status')=='후보' and hypothesis.get('nodes'):
            output=dict(hypothesis)
            output['requested_order']=requested_order
            output['auto_recounted']=(level!=requested_order)
            output['recount_scales_tried']=priority[:priority.index(level)+1]
            output['previous_rejections']=discarded
            if level!=requested_order:
                output['notes'] += (f' 최초 선택 민감도 {requested_order}에서 활성 파동을 확인하지 못해 '
                                   f'민감도 {level}의 규칙 통과 후보로 자동 재탐색했습니다. '
                                   '서로 다른 민감도의 파동은 동일한 파동 차수가 아니며 확률 순위가 아닙니다.')
            return output
        discarded.append({'order':level,'notes':hypothesis.get('notes',''),
                          'invalidation_event':hypothesis.get('invalidation_event')})
    output=dict(first or {})
    output['requested_order']=requested_order
    output['auto_recounted']=False
    output['recount_scales_tried']=priority
    output['previous_rejections']=discarded
    output['notes']=(output.get('notes','')+' 3·5·8·12봉으로 재탐색했지만 '
                     '확인 가능한 유효한 파동이 없습니다. 임의 번호를 생성하지 않습니다.')
    return output
