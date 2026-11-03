"""Offline deployment dependency preflight, avoiding 10.9's missing-module failure.

Does not use credentials, network, broker API or market data. Intended for CLI
and a fast check before scheduled jobs. This is NOT a runtime smoke test.
"""
from __future__ import annotations
from pathlib import Path
import ast
import json
import sys

CORE = (
    # All locally imported runtime modules for the Streamlit app and GitHub job.
    # This catches missing-module deployments BEFORE 15min data collection.
    'app.py','alerts.py','archive_collector.py','auto_learning.py',
    'chart_patterns.py','context_model.py','crypto_intraday.py',
    'crypto_transfer.py','elliott_waves.py','engine.py','explain_chat.py',
    'forecast_forward.py','historical_backfill.py','intelligence.py',
    'journal.py','local_ai.py','market_data_guard.py','market_decision.py',
    'mobile_ui.py','opportunity_diagnostics.py','options_engine.py',
    'paper_audit.py','pattern_discovery.py','point_in_time.py',
    'preflight.py','research.py','risk_engine.py','scheduled_scan.py',
    'sec_backfill.py','sec_history.py','stock_data_provider.py','strict_validation.py','timeframe.py',
    'v8_features.py','analysis_decision.py','wave_auto_recount.py',
    'quality_guard.py','historical_evidence.py','wave_bottoms.py','wave_bottom_audit.py','system_check.py',
)


def check(root:Path)->dict:
    root=Path(root)
    missing=[name for name in CORE if not (root/name).is_file()]
    syntax_errors=[]
    for name in CORE:
        p=root/name
        if p.is_file():
            try:ast.parse(p.read_text(encoding='utf-8'),filename=name)
            except (SyntaxError,UnicodeError) as e:syntax_errors.append(f'{name}: {str(e)[:130]}')
    cfg=root/'config'/'watchlist.txt'
    if not cfg.is_file():missing.append('config/watchlist.txt')
    if not (root/'requirements.txt').is_file():missing.append('requirements.txt')
    # Detect newly imported local modules that have been omitted from CORE.
    # Imports from Python stdlib and installed dependencies are excluded.
    third_party={'numpy','pandas','sklearn','scipy','plotly','streamlit','yfinance','requests','pytest'}
    stdlib=getattr(sys,'stdlib_module_names',set()) | {'__future__'}
    for name in CORE:
        f=root/name
        if not f.is_file():continue
        try:tree=ast.parse(f.read_text(encoding='utf-8'))
        except (SyntaxError,UnicodeError):continue
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and node.level==0:
                mods=[node.module.split('.')[0]]
            elif isinstance(node,ast.Import):
                mods=[alias.name.split('.')[0] for alias in node.names]
            else:continue
            for mod in mods:
                if mod in third_party or mod in stdlib or mod.startswith('_'):continue
                if (root/(mod+'.py')).is_file() or (root/mod).is_dir():continue
                item=f'{name}: import {mod}.py (파일 없음)'
                if item not in missing:missing.append(item)
    status='ok' if not missing and not syntax_errors else 'failed'
    return {'status':status,'missing':missing,'syntax_errors':syntax_errors,
            'files_checked':len(CORE),'note':'정적 파일 점검. 실제 수집·학습 성공을 뜻하지 않음'}


def main():
    state=check(Path(__file__).resolve().parent)
    print('SMARTSTOCK_PREFLIGHT '+json.dumps(state,ensure_ascii=False))
    if state['status']!='ok':
        print('11.2 통합 업데이트 ZIP의 프로그램 파일을 같은 폴더에 설치하고 다시 실행하세요.',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
