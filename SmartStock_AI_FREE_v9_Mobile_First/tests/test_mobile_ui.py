from mobile_ui import mobile_page, resolve_known_ticker

KNOWN={'CRWV':'코어위브','NVDA':'엔비디아','TSLA':'테슬라','AAPL':'애플',
       '005930.KS':'삼성전자','000660.KS':'SK하이닉스'}


def test_main_navigation():
    assert mobile_page('📈 차트') == 'chart'
    assert mobile_page('🔮 예측') == 'forecast'
    assert mobile_page('📌 옵션') == 'options'


def test_mobile_extra_sections():
    extras={
        '🛡 위험관리':'risk', '📰 뉴스':'news', '🏢 재무·시장':'fund',
        '🧩 AI 종합 해설':'explain','🧠 학습·검증':'model','🔎 관심종목 비교':'quick'
    }
    for label,page in extras.items():
        assert mobile_page('☰ 더보기',label)==page


def test_missing_subpage_falls_back_to_risk():
    assert mobile_page('☰ 더보기')=='risk'
    assert mobile_page('☰ 더보기','존재하지 않는 메뉴')=='risk'


def test_known_ticker_exact():
    assert resolve_known_ticker('crwv',KNOWN)=='CRWV'
    assert resolve_known_ticker(' NVDA ',KNOWN)=='NVDA'


def test_korean_name_exact():
    assert resolve_known_ticker('코어위브',KNOWN)=='CRWV'
    assert resolve_known_ticker('삼성전자',KNOWN)=='005930.KS'


def test_korean_numeric_code():
    assert resolve_known_ticker('005930',KNOWN)=='005930.KS'
    assert resolve_known_ticker('000660.KS',KNOWN)=='000660.KS'
    assert resolve_known_ticker('123456.KQ',KNOWN)=='123456.KQ'


def test_dont_guess_company_name_as_ticker():
    assert resolve_known_ticker('Tesla Motors',KNOWN) is None
    assert resolve_known_ticker('MICROSOFT',KNOWN) is None
    assert resolve_known_ticker('테슬라 주식',KNOWN) is None


def test_unsafe_input_not_used_as_ticker():
    assert resolve_known_ticker('NVDA<script>',KNOWN) is None
    assert resolve_known_ticker('../secret',KNOWN) is None
