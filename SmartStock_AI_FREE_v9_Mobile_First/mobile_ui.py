"""Deterministic and testable routing/search helpers for the mobile-first UI.
No network requests, Streamlit state, or side effects.
"""
from __future__ import annotations
import re

PAGE_MAP={
    '📈 차트':'chart',
    '🔮 예측':'forecast',
    '📌 옵션':'options',
}
MORE_MAP={
    '🛡 위험관리':'risk',
    '📰 뉴스':'news',
    '🏢 재무·시장':'fund',
    '🧩 AI 종합 해설':'explain',
    '🧠 학습·검증':'model',
    '🔎 관심종목 비교':'quick',
}

def mobile_page(tab: str, detail: str | None = None) -> str:
    """Route four simple big tabs and six drill-down pages."""
    if tab == '☰ 더보기':
        return MORE_MAP.get(detail or '🛡 위험관리','risk')
    return PAGE_MAP.get(tab,'chart')


def resolve_known_ticker(search: str, popular: dict[str,str]) -> str | None:
    """Use only unambiguous tickers/known aliases; don't guess from names.

    Generic unknown English words must go through symbol search, rather than
    being interpreted as genuine tickers (e.g. Tesla -> TESLA is wrong).
    """
    term=search.strip()
    upper=term.upper()
    if upper in popular: return upper
    exact=[sym for sym,name in popular.items() if term.casefold()==name.casefold()]
    if len(exact)==1:return exact[0]
    if re.fullmatch(r'\d{6}(\.(KS|KQ))?',upper):
        return upper if '.' in upper else upper+'.KS'
    if re.fullmatch(r'[A-Z]{1,5}\.[A-Z]{1,4}',upper):
        return upper
    return None
