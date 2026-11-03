"""SmartStock AI: safe, reproducible scenario adjudication.

This module NEVER invents probabilities.  Validated scenario ranking is always
available when at least one rule-valid/data-valid candidate exists.  Numeric
outcome probabilities are released only when comparable, non-overlapping,
out-of-sample historical cases have matured and passed support thresholds.

Expected candidate:
    id, title, valid (bool), data_valid (bool), outcome (optional:
    '상승'/'하락'/'횡보'), rank_score (optional non-probability 0..100),
    invalidation (optional), evidence (optional list of strings)

Expected context:
    symbol, timeframe, horizon_bars, signal_bucket, model_version, asof (ISO
    timestamp with UTC offset).  Numeric probabilities are *not* available for
    wave-shape-only alternatives that have no directly testable future outcome.

Expected historical case (all required for a usable probability sample):
    event_id, symbol, timeframe, horizon_bars, signal_bucket, model_version,
    prediction_time, resolved_time (ISO aware), actual_outcome,
    split='out_of_sample', non_overlapping=True.

Important: this is an integration module, not a live market data feed or an
independent investment recommendation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from math import sqrt, isfinite
from typing import Any, Iterable, Mapping

OUTCOMES = ('상승', '하락', '횡보')


def _aware_time(value: Any) -> datetime | None:
    if not isinstance(value, (str, datetime)):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
        return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _numeric_score(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        score = float(value)
        return score if isfinite(score) and 0 <= score <= 100 else None
    except (ValueError, TypeError):
        return None


def _clean_candidates(candidates: Iterable[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    kept, discarded, seen = [], [], set()
    for i, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            discarded.append({'id': f'item-{i}', 'reason': '후보 형식 오류'})
            continue
        key = str(item.get('id') or f'candidate-{i}')
        if key in seen:
            discarded.append({'id': key, 'reason': '동일 후보 중복'})
            continue
        seen.add(key)
        if item.get('valid') is not True or item.get('data_valid') is not True:
            discarded.append({'id': key, 'reason': str(item.get('reject_reason') or '분석 규칙 또는 데이터 검증 실패')})
            continue
        cleaned = {'id': key, 'title': str(item.get('title') or key),
                   'outcome': item.get('outcome') if item.get('outcome') in OUTCOMES else None,
                   'rank_score': _numeric_score(item.get('rank_score')),
                   'evidence': [str(e) for e in (item.get('evidence') or [])][:8],
                   'invalidation': item.get('invalidation'),
                   'origin': str(item.get('origin') or '기본 분석'),
                   'input_order': i}
        kept.append(cleaned)
    return kept, discarded


def _historical_outcomes(records: Iterable[Mapping[str, Any]], context: Mapping[str, Any]) -> list[str]:
    """Filter out future labels, in-sample results, pooled regimes and duplicates.

    `non_overlapping` must be computed upstream using horizon-based purging,
    not guessed here. Only comparable past resolved events may be used.
    """
    now = _aware_time(context.get('asof'))
    if now is None:
        return []
    required = ('symbol', 'timeframe', 'horizon_bars', 'signal_bucket', 'model_version')
    if any(context.get(k) in (None, '') for k in required):
        return []
    eligible, seen = [], set()
    for r in records:
        if not isinstance(r, Mapping):
            continue
        if any(r.get(k) != context.get(k) for k in required):
            continue
        if r.get('split') != 'out_of_sample' or r.get('non_overlapping') is not True:
            continue
        outcome = r.get('actual_outcome')
        if outcome not in OUTCOMES:
            continue
        predicted, resolved = _aware_time(r.get('prediction_time')), _aware_time(r.get('resolved_time'))
        if predicted is None or resolved is None or not (predicted < resolved <= now):
            continue
        event_id = str(r.get('event_id') or '')
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        eligible.append((predicted,resolved,outcome,event_id))
    # Independently verify spacing and avoid counting overlapping forecast
    # windows twice, even if a malformed provider marks all rows as independent.
    accepted=[]
    last_resolved=None
    for predicted,resolved,outcome,event_id in sorted(eligible,key=lambda r:(r[0],r[1],r[3])):
        if last_resolved is not None and predicted < last_resolved:
            continue
        accepted.append(outcome)
        last_resolved=resolved
    return accepted


def _wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0., 1.)
    p = wins / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    margin = z*sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return (max(0., center-margin), min(1., center+margin))


def decide(
    candidates: Iterable[Mapping[str, Any]],
    context: Mapping[str, Any],
    historical_cases: Iterable[Mapping[str, Any]] = (),
    additional_searches: Iterable[tuple[str, Iterable[Mapping[str, Any]]]] = (),
    min_samples: int = 80,
    max_interval_width: float = 0.26,
) -> dict[str, Any]:
    """Re-search, then rank; never substitute a heuristic score for probability.

    Additional searches are finite, upstream-generated alternatives such as
    another Elliott sensitivity, timeframe or market-regime hypothesis. They
    are evaluated even if the initial search already found a candidate.
    """
    combined = list(candidates)
    searched = ['기본 분석']
    for origin, extra in list(additional_searches)[:8]:
        searched.append(str(origin))
        for item in extra:
            if isinstance(item, Mapping):
                enriched = dict(item)
                enriched['origin'] = str(origin)
                combined.append(enriched)
            else:
                combined.append(item)
    valid, rejected = _clean_candidates(combined)
    result: dict[str, Any] = {
        'status': '재분석 필요', 'top': None, 'alternatives': [],
        'rejected': rejected, 'searched': searched, 'probabilities': None,
        'probability_basis': '검증 불충분', 'sample_count': 0, 'confidence_intervals': None,
        'notice': '유효한 분석 후보가 없으면 근거 없는 정답이나 확률을 만들지 않습니다.',
        'symbol': context.get('symbol'), 'timeframe': context.get('timeframe'),
        'horizon_bars': context.get('horizon_bars'), 'asof': context.get('asof'),
    }
    if not valid:
        return result

    samples = _historical_outcomes(historical_cases, context)
    result['sample_count'] = len(samples)
    counts = Counter(samples)
    has_testable_outcome = any(c['outcome'] in OUTCOMES for c in valid)
    if has_testable_outcome and len(samples) >= max(1, int(min_samples)):
        intervals = {name: _wilson_interval(counts[name], len(samples)) for name in OUTCOMES}
        # Precision gate controls *all* scenario values, not just the winner.
        if max(hi - lo for lo, hi in intervals.values()) <= max_interval_width:
            probs = {name: counts[name] / len(samples) for name in OUTCOMES}
            result['probabilities'] = probs
            result['confidence_intervals'] = intervals
            result['probability_basis'] = '동일 조건의 시간순 검증표본 관찰 빈도 (미래 보장 아님)'

    def rank(c: dict) -> tuple[float, float, int]:
        p = result['probabilities']
        valid_prob = p is not None and c['outcome'] is not None
        # With no usable probabilities, model rank scores are explicitly NOT probabilities.
        return (p[c['outcome']] if valid_prob else -1.0,
                c['rank_score'] if c['rank_score'] is not None else -1.0,
                -c['input_order'])

    valid.sort(key=rank, reverse=True)
    result['top'] = valid[0]
    result['alternatives'] = valid[1:4]
    result['status'] = '확률 검증 후보' if result['probabilities'] is not None and valid[0]['outcome'] else '유력 후보 (확률 미검증)'
    if result['probabilities'] is not None:
        result['notice'] = ('표시한 %는 동일 종목·시간봉·예측기간·신호 조건의 과거 관찰비율입니다. '
                            '파동 해석의 정답 확률도, 미래 수익 보장도 아닙니다.')
    else:
        result['notice'] = ('규칙을 통과한 최상위 후보를 표시합니다. '
                            '근거가 부족해 임의의 성공확률 %는 붙이지 않습니다. '
                            '분석 점수는 실제 예측 확률과 다릅니다.')
    return result


def wave_candidates(wave: Mapping[str, Any]) -> list[dict]:
    """Keep Elliott wave probabilities unavailable until a testable forecast exists.

    The earlier recount's first entry is a rule-valid *interpretation*, not a
    quantified future event.  It must not gain an invented 90% label.
    """
    if wave.get('status') != '후보' or not wave.get('nodes'):
        return []
    nodes = wave['nodes']
    title = f"{wave.get('phase', '엘리엇 파동')} · 확정 {nodes[-1].get('파동')}파"
    return [{'id': 'elliott-primary', 'title': title, 'valid': True,
             'data_valid': True, 'outcome': None, 'rank_score': None,
             'origin': f"피벗 민감도 {wave.get('order', '?')}",
             'invalidation': wave.get('invalidation'),
             'evidence': [str(wave.get('notes', ''))]}]


def render_streamlit_report(report: Mapping[str, Any]) -> None:
    """Galaxy-friendly, auditable candidate explanations and separate probabilities."""
    import streamlit as st
    st.subheader('🧭 AI 재검증 분석')
    top=report.get('top')
    if not top:
        st.warning('현재 규칙과 데이터 품질을 통과한 후보가 없습니다. 번호나 방향을 만들어내지 않습니다.')
        st.caption('재탐색 범위: '+', '.join(report.get('searched',[])))
        return
    st.markdown(f"**현재 1순위: {top['title']}**")
    probs=report.get('probabilities')
    if probs is not None and top.get('outcome'):
        outcome=top['outcome']
        interval=report['confidence_intervals'][outcome]
        st.metric(f'{report.get("horizon_bars")}봉 후 {outcome} 과거 관찰 빈도',f'{probs[outcome]*100:.1f}%')
        st.caption(f'동일 조건의 과거 검증표본 {report["sample_count"]}건 · 95% Wilson 구간 {interval[0]*100:.1f}~{interval[1]*100:.1f}%')
        st.caption('과거 빈도: '+' · '.join(f'{name} {probs[name]*100:.1f}%' for name in OUTCOMES))
        st.caption('이 수치는 실제 미래의 확정 확률이 아니며, 아래 규칙 일치도 점수와 다릅니다.')
    else:
        st.caption('검증된 숫자 확률 없음 · 표본 수와 95% 구간 폭 기준을 충족하지 못함')
    if top.get('rank_score') is not None:
        st.metric('1순위 규칙 일치도 (확률 아님)',f'{top["rank_score"]:.1f}점')
        st.caption('계열별 합의도 분배 점수입니다. 세 시나리오 점수의 합이 100이어도 발생확률이 아닙니다.')
    if top.get('invalidation'):
        st.info('**1순위 재검토 가격 기준**: '+str(top['invalidation']))
    with st.expander('📊 1순위 점수 계산과 실측 근거',expanded=False):
        for e in top.get('evidence',[]):st.caption('• '+e)
        if not top.get('evidence'):st.caption('독립적으로 표시할 수 있는 상세 지표 근거가 없습니다.')
    with st.expander('대안 시나리오와 제외된 분석',expanded=False):
        for item in report.get('alternatives',[]):
            st.markdown(f"**{item['title']}**")
            if item.get('rank_score') is not None:
                st.caption(f'규칙 일치도 {item["rank_score"]:.1f}점 (예측 확률 아님)')
            if item.get('invalidation'):
                st.caption('재검토 기준: '+str(item['invalidation']))
            for e in item.get('evidence',[])[:6]:st.caption('• '+e)
        for item in report.get('rejected',[])[:15]:
            st.caption(f"• 제외: {item['id']} — {item['reason']}")
    st.caption(report.get('notice',''))
